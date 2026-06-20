# Restormer Denoising 
## The Objective : 
Raw thermal data (Band 10) is fundamentally noisy due to atmospheric scattering and sensor hardware limitations. If we feed raw noise directly into a Super-Resolution model (like SwinIR), the network will hallucinate and "super-resolve" the noise into permanent, sharp artifacts.

The Restormer acts as an intelligent, non-destructive filter. Its sole job is to ingest the noisy ```(1, 256, 256)``` IR tensor, strip away the environmental sensor noise, and output a clean ```(1, 256, 256)``` IR tensor containing only true structural signatures.

## Restormer
Standard Vision Transformers (ViTs) divide an image into patches and compute self-attention across the spatial dimension. For a $256 \times 256$ image, the spatial dimension is massive, leading to an incredibly heavy computational cost of $O(N^2)$ (where $N$ is the number of pixels).
Restormer flips this math on its head. It computes self-attention across the channel dimension rather than the spatial dimension, changing the computational cost to $O(C^2)$ (where $C$ is the number of channels). This allows it to process high-resolution images globally with a fraction of the VRAM.

## The Architecture
Consists of 2 highly specilaized PyTorch Classes

### Multi-Dconv Head Transposed Attention (MDTA)
Purpose: This is the "brain" of the Restormer. It looks at the entire image globally to understand what is actual thermal structure (road,building,etc) and what is random sensor noise.

Mechanism : It uses depth-wise convolutions to aggregate local pixel context, and then calculates an attention map across the channels to figure out which features to keep and which to suppress.

### Gated-Dconv Feed-Forward Network (GDFN)
Purpose: This acts as the "refinery." Once the MDTA identifies the noise, the GDFN safely removes it without blurring the sharp edges of the thermal features.

Mechanism: It uses a gating mechanism (like a mathematical valve) that multiplies two parallel pathways together, effectively shutting off the flow of noisy pixels while allowing clean structural pixels to pass through.

## Inplementation Plan : 
* step 1 : Code the ```MDTA``` (Attention) and ```GDF``` (Feed-Forward) classes.
* Step 2 : Combine them into a single ```TransformerBlock```
* Step 3 : Wrap those blocks into the final ```Restormer``` module that takes our 1-channel IR input and outputs the 1-channel clean IR output