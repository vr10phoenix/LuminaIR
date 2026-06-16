#Restormer
class Restormer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=48, num_blocks=4, num_heads=8):
        super(Restormer, self).__init__()
        
        # Feature Embedding
        self.embed_conv = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)
        
        # processing engine
        self.blocks = nn.ModuleList([
            TransformerBlock(channels=dim, num_heads=num_heads) for _ in range(num_blocks)
        ])
        
        # Final Mapping (Projects deep features back into a 1-channel thermal image)
        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # Save the original raw image for the Global Residual
        identity = x
        
        # Extract initial features
        fea = self.embed_conv(x)
        
        # Pass sequentially through the blocks
        for block in self.blocks:
            fea = block(fea)
            
        # Project back to exactly 1 channel
        out = self.mapping(fea)
        
        # Global Residual Connection: Add back the structural baseline
        out = out + identity
        
        return out