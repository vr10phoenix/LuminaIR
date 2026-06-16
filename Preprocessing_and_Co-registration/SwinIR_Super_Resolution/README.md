# SwinIR Engine

## Local Window Attention
standard spatial Transformer attention on a 256x256 image, the math requires comparing every single pixel to every other pixel ($65,536 \times 65,536$ operations)

### The Swin Solution:
 SwinIR slices your image into tiny, non-overlapping windows (usually 8x8 pixels). It then strictly calculates attention only within that specific 8x8 window.

The Result: The computational cost drops from astronomical to perfectly linear, making it incredibly fast and VRAM-friendly.

## Shifted Windows
In the very next layer, SwinIR mathematically shifts the 8x8 grid over by half a window (4 pixels).

The Result: The new windows now overlap the boundaries of the old windows. This allows the network to pass information across the entire image, perfectly stitching the thermal structures together without blocky artifacts.

## The Upsampling Module (PixelShuffle)
nce the Shifted Windows have extracted ultra-deep, hyper-sharp features from the thermal map, we need to physically increase the resolution of the image (Super-Resolution) before passing it to the colorization phase.

SwinIR uses a Sub-Pixel Convolution (often called PixelShuffle). It generates massive numbers of feature channels and then physically rearranges them into spatial pixels.The Result: The spatial resolution is doubled (e.g., 256x256 $\rightarrow$ 512x512) with mathematical sharpness, creating high-definition edges for your thermal data.

## Implementation plan : 
* Step 1: The Core Swin Layers. We will code the mathematical functions that slice the image into windows, calculate attention, and shift the grid.

* Step 2: The Residual Swin Transformer Block (RSTB). We will stack those window layers together with residual connections to create deep feature extractors.

* Step 3: The Final SwinIR Wrapper. We will combine the feature extractors with the PixelShuffle upsampler to take our 1-channel clean IR image and output a high-resolution 1-channel sharp IR image.