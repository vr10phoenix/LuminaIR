import torch
import torch.nn as nn
import torch.nn.functional as F

#Multi-Dconv Head Transposed Attention (MDTA)
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 Conv to generate Q, K, V
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        # 3x3 Depth-wise Conv to aggregate local pixel context 
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, 
                                    stride=1, padding=1, groups=channels * 3, bias=False)
        
        # 1x1 Conv to project back original channel dimension
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Generate Q, K, V with local context
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Rearrange to compute attention across channels, NOT spatial pixels
        q = q.view(b, self.num_heads, -1, h * w)
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)

        # L2 Normalization for stability
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # The transposed attention map (Channel-wise)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Apply attention to V and reshape
        out = (attn @ v)
        out = out.view(b, c, h, w)
        
        out = self.project_out(out)
        return out

# Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        
        # Expand channels
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        
        # Local structural feature extraction
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, 
                                stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        
        # Reduce channels
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        
        # The Gating Mechanism
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # Only clean structural pixels pass through
        
        x = self.project_out(x)
        return x