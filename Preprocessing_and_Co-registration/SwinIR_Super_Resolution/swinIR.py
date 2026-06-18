import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    """
    Slices the full image into isolated, non-overlapping windows.
    """
    B, H, W, C = x.shape
    #isolate the grid
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Stitches the isolated windows perfectly back together into the full image.
    Input: [Batch * num_windows, window_size, window_size, Channels] , Output: [Batch, Height, Width, Channels]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# Local Window Attention
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Q, K, V

        #Attention calculation
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # apply Softmax
        attn = self.softmax(attn)

        # apply attention to Values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # Final projection
        x = self.proj(x)
        return x
    

    # The Multi-Layer Perceptron (mlp)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

#Swin Transformer 
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Normalization Layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.attn = WindowAttention(dim, window_size, num_heads)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4.0))

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x

        x = self.norm1(x)

        if self.shift_size > 0:
            # Shift image by 4 pixels diagonally
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        #Slice the image into 8x8 windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Pass through the Attention Brain
        attn_windows = self.attn(x_windows)

        # convert back to full image
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
    

#Residual Swin Transformer 
class RSTB(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, depth=6):
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2
            )
            for i in range(depth)
        ])

        # The final Convolution layer
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # x shape: [Batch, Height, Width, Channels]
        shortcut = x

        for block in self.blocks:
            x = block(x)

        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()

        # apply filter
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x + shortcut

        return x
    
# The PixelShuffle Upsampler
class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_dim, out_channels=1, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        expanded_dim = in_dim * (upscale_factor ** 2)
        self.expand_conv = nn.Conv2d(in_dim, expanded_dim, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self.final_conv = nn.Conv2d(in_dim, out_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        x = self.expand_conv(x)
        x = self.pixel_shuffle(x)
        x = self.lrelu(x)
        x = self.final_conv(x)

        return x
    

# Wrapper function
class SwinIR(nn.Module):
    def __init__(self, in_channels=1, dim=48, num_heads=8, window_size=8, depths=[6, 6, 6, 6], upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.conv_first = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.layers = nn.ModuleList()
        for depth in depths:
            self.layers.append(
                RSTB(dim=dim, num_heads=num_heads, window_size=window_size, depth=depth)
            )

        self.conv_after_body = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.upsampler = PixelShuffleUpsampler(in_dim=dim, out_channels=1, upscale_factor=upscale_factor)

    def forward(self, x):
        x_first = self.conv_first(x)
        res = x_first.permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            res = layer(res)

        res = res.permute(0, 3, 1, 2).contiguous()
        res = self.conv_after_body(res)
        res = res + x_first

        out = self.upsampler(res)
        return out