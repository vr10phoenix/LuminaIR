import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- MDTA ----------
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3,
                                    stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out

# ---------- GDFN ----------
class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# ---------- Transformer Block ----------
class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor=2.66):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        identity = x
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()
        x = self.attn(x_norm)
        x = x + identity

        identity = x
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()
        x = self.ffn(x_norm)
        x = x + identity
        return x

# ---------- Restormer Model ----------
class Restormer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=48, num_blocks=4, num_heads=8):
        super(Restormer, self).__init__()
        self.embed_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(channels=dim, num_heads=num_heads) for _ in range(num_blocks)
        ])
        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        identity = x
        fea = self.embed_conv(x)
        for block in self.blocks:
            fea = block(fea)
        out = self.mapping(fea)
        out = out + identity
        return out

# ---------- Pre-trained weights loader (converts 3‑channel to 1‑channel) ----------
def pre_restormer(pretrained_weights_path=None, device="cuda"):
    """
    Instantiates a Restormer, loads 3-channel pre-trained weights,
    modifies the first layer for 1-channel TIR inputs.
    """
    model = Restormer(in_channels=3, out_channels=1, dim=48, num_blocks=4, num_heads=8)

    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading Pre-Trained Weights from {pretrained_weights_path}...")
        checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['params'] if 'params' in checkpoint else checkpoint, strict=False)
    else:
        print("No pre-trained weights provided or file not found. Starting from scratch.")

    # Convert first conv layer from 3 channels to 1 channel
    first_layer_weights = model.embed_conv.weight.data
    single_channel_weights = torch.mean(first_layer_weights, dim=1, keepdim=True)
    new_first_layer = nn.Conv2d(in_channels=1,
                                out_channels=48,
                                kernel_size=3,
                                padding=1,
                                bias=False)
    new_first_layer.weight.data = single_channel_weights
    model.embed_conv = new_first_layer

    return model.to(device)