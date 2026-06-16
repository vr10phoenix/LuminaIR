class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor=2.66):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        #Attention Pathway
        identity = x
        
        # Permute for LayerNorm: [B, C, H, W] :--> [B, H, W, C]
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_norm)

        # Permute back: [B, H, W, C] :--> [B, C, H, W]
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        x = self.attn(x_norm)
        x = x + identity  # Residual Connection 1
        
        # Feed-Forward Pathway
        identity = x
        
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        
        x = self.ffn(x_norm)
        x = x + identity  # Residual Connection 2
        
        return x