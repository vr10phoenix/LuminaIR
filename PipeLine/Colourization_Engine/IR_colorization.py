import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO
import torch.optim as optim
import time
from Restormer_Denoising.pytorch_dataloader import train_loader

class SemanticGuardrail(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        ).to(self.device)


        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self._freeze_weights()
        self.eval()

    def _pad_to_multiple(self, x, divisor=14):
        """Pads tensor to the nearest multiple of 14."""
        h, w = x.shape[-2:]
        pad_h = (divisor - (h % divisor)) % divisor
        pad_w = (divisor - (w % divisor)) % divisor
        if pad_h > 0 or pad_w > 0:
            # Pad bottom and right
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (h, w) # Return input and original dimensions for cropping later

    def _freeze_weights(self):
        for param in self.segformer.parameters(): param.requires_grad = False
        for param in self.dinov2.parameters(): param.requires_grad = False

    def forward(self, clean_ir_batch):
        # Dynamic Padding
        padded_ir, (orig_h, orig_w) = self._pad_to_multiple(clean_ir_batch, 14)
        ir_3ch = padded_ir.repeat(1, 3, 1, 1)

        with torch.no_grad():
            # Segformer
            seg_outputs = self.segformer(ir_3ch)
            logits = F.interpolate(
                seg_outputs.logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )
            semantic_mask = logits.argmax(dim=1).unsqueeze(1)

            #DINO v2
            dino_features = self.dinov2.forward_features(ir_3ch)['x_norm_patchtokens']

        return semantic_mask, dino_features

guardrail = SemanticGuardrail().cuda()



#  Semantic Fusion Layer 
class SemanticFusionLayer(nn.Module):
    def __init__(self, mask_channels=1, ir_channels=1, fused_channels=64):
        super().__init__()
        """
        Takes the 1-channel thermal image and 1-channel argmax mask and fuses them.
        Using shallow convolutions instead of one-hot encoding saves massive VRAM.
        """
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(mask_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True)
        )

        # Shallow feature extraction
        self.ir_encoder = nn.Sequential(
            nn.Conv2d(ir_channels, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True)
        )

        # Fusion Block
        self.fusion = nn.Sequential(
            nn.Conv2d(32, fused_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(fused_channels),
            nn.ReLU(True)
        )

    def forward(self, clean_ir, semantic_mask):
        semantic_mask = semantic_mask.float()

        ir_feat = self.ir_encoder(clean_ir)
        mask_feat = self.mask_encoder(semantic_mask)

        fused = torch.cat([ir_feat, mask_feat], dim=1)
        return self.fusion(fused)



# Pix2PixHD Generator
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        # residual connection 
        return x + self.conv_block(x)

class GlobalGenerator(nn.Module):
    def __init__(self, input_channels=64, output_channels=3, n_residual_blocks=6):
        super().__init__()
        """
        The heavy-lifting convolutional network acts as the Painter.
        """
        # Downsampling 
        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1), # Drops to 256x256
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Drops to 128x128
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )

        #Semantic Processing
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(256))
        self.residuals = nn.Sequential(*res_blocks)

        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Back to 256x256
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Back to 512x512
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, kernel_size=7),
            nn.Tanh() # Strictly locks the output to a mathematical range of [-1, 1]
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.residuals(x)
        x = self.upsample(x)
        return self.output_layer(x)



class SemanticGuidedGenerator(nn.Module):
    def __init__(self, guardrail_module):
        super().__init__()
        """
        Wrapper for the Pix2PixHD Generator.
        Combines the frozen Semantic Guardrail with our active generative layers.
        """
        # Pre-trained stencil maker
        self.guardrail = guardrail_module

        # Data compressor
        self.fusion = SemanticFusionLayer(mask_channels=1, ir_channels=1, fused_channels=64)

        # The Painter
        self.generator = GlobalGenerator(input_channels=64, output_channels=3, n_residual_blocks=6)

    def forward(self, clean_ir_batch):
        """
        Input: clean_ir_batch [16, 1, 512, 512] 
        Returns: fake_rgb [16, 3, 512, 512]
        """
        semantic_mask, dino_features = self.guardrail(clean_ir_batch)
        fused_tensor = self.fusion(clean_ir_batch, semantic_mask)
        fake_rgb = self.generator(fused_tensor)

        return fake_rgb, semantic_mask

pix2pix_generator = SemanticGuidedGenerator(guardrail).cuda()


# PatchGAN critic
class PatchGAN_Discriminator(nn.Module):
    def __init__(self, input_channels=4, base_filters=64, n_layers=3):
        super().__init__()
        """
        Critiques a specific scale of the image.
        Input channels: 3 (RGB) + 1 (Semantic Mask) = 4.
        """
        # Input layer
        sequence = [
            nn.Conv2d(input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        # Feature layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(base_filters * nf_mult_prev, base_filters * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(base_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(base_filters * nf_mult_prev, base_filters * nf_mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output layer
        sequence += [
            nn.Conv2d(base_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

# multi-scale Wrapper 
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels=4, num_scales=2):
        super().__init__()
        """
        The Master Critic. Runs two identical PatchGANs at different image resolutions.
        D_1 looks at 512x512 (Micro textures).
        D_2 looks at 256x256 (Global coherence).
        """
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for _ in range(num_scales):
            self.discriminators.append(PatchGAN_Discriminator(input_channels=input_channels))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        critiques = []
        for i in range(self.num_scales):
            critiques.append(self.discriminators[i](x))

            if i != self.num_scales - 1:
                x = self.downsample(x)

        return critiques

# initialize the master critic
master_critic = MultiScaleDiscriminator(input_channels=4).cuda()


# loss engine LSGAN & L1
class MasterLossEngine(nn.Module):
    def __init__(self, lambda_l1=10.0):
        super().__init__()
        """
        mathematical adjudicator between the Generator and the Critic.
        """
        # LSGAN loss 
        self.adversarial_loss = nn.MSELoss()
        # L1 loss 
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def forward(self, critiques_fake, fake_rgb, real_rgb):
        loss_G_adv = 0.0

        # multi-scale adversarial Loss
        for critique in critiques_fake:
            target_real = torch.ones_like(critique, device=critique.device)
            loss_G_adv += self.adversarial_loss(critique, target_real)

        # calculate tructural L1 Loss
        loss_G_l1 = self.l1_loss(fake_rgb, real_rgb) * self.lambda_l1
        # total generator loss
        total_G_loss = loss_G_adv + loss_G_l1

        return total_G_loss, loss_G_adv, loss_G_l1

# initialize the loss engine
loss_engine = MasterLossEngine().cuda()
print("[SUCCESS] Master Loss Engine initialized and deployed to GPU.")


# YOLO
class YOLOv11Validator(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        """
        The Downstream Machine Vision Validator.
        Tests if the Generator's RGB output is actually readable by SOTA AI.
        """
        print("Initializing YOLO11-Nano Downstream Validator...")
        self.yolo = YOLO("yolo11n.pt")
        self.yolo.to(device)
        self._freeze_weights()
        self.yolo.model.eval()

    def _freeze_weights(self):
        for param in self.yolo.model.parameters():
            param.requires_grad = False

    def forward(self, fake_rgb):
        with torch.no_grad():
            yolo_ready_images = (fake_rgb + 1.0) / 2.0
            results = self.yolo.predict(yolo_ready_images , verbose = False , stream = False)

        return results

# initialize the validator
validator = YOLOv11Validator().cuda()



# Standard GAN learning rate
lr = 0.0002

# generator Optimizer 
optimizer_G = optim.Adam(
    list(pix2pix_generator.fusion.parameters()) + list(pix2pix_generator.generator.parameters()),
    lr=lr, betas=(0.5, 0.999)
)

# discriminator optimizer
optimizer_D = optim.Adam(master_critic.parameters(), lr=lr, betas=(0.5, 0.999))

print("[SUCCESS] Optimizers initialized and locked to their respective networks.")

# training hyperparameters
epochs = 50
print_freq = 50 # Print loss every 50 batches
mse_loss = nn.MSELoss()

# Training Loop
for epoch in range(epochs):
    epoch_start_time = time.time()

    for i, (clean_ir, real_rgb) in enumerate(train_loader):
        clean_ir = clean_ir.cuda()
        real_rgb = real_rgb.cuda()
        
        #forward pass
        fake_rgb, semantic_mask = pix2pix_generator(clean_ir)
        fake_concat = torch.cat([fake_rgb.detach(), semantic_mask], dim=1)
        real_concat = torch.cat([real_rgb, semantic_mask], dim=1)

        optimizer_D.zero_grad()
        
        # critic for fake image
        critiques_fake = master_critic(fake_concat)
        loss_D_fake = sum([mse_loss(critique, torch.zeros_like(critique)) for critique in critiques_fake])

        # critik for real image
        critiques_real = master_critic(real_concat)
        loss_D_real = sum([mse_loss(critique, torch.ones_like(critique)) for critique in critiques_real])

        # backpropagate discriminator
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # generator updation
        optimizer_G.zero_grad()

        fake_concat_for_G = torch.cat([fake_rgb, semantic_mask], dim=1)
        critiques_for_G = master_critic(fake_concat_for_G)

        loss_G, loss_G_adv, loss_G_l1 = loss_engine(critiques_for_G, fake_rgb, real_rgb)

        # backpropagate generator
        loss_G.backward()
        optimizer_G.step()

        # logging and downstream validation
        if i % print_freq == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                  f"[G L1: {loss_G_l1.item():.4f}]")


            valid_image = ((fake_rgb[0:1] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            yolo_results = validator(valid_image)


            detections = len(yolo_results[0].boxes)
            print(f"   -> YOLOv11 Validator spotted {detections} objects in generated image.")

    print(f"End of Epoch {epoch} - Time taken: {time.time() - epoch_start_time:.2f} seconds")

    # save model checkpoints strictly every 5 epochs t
    if epoch % 5 == 0:
        torch.save(pix2pix_generator.state_dict(), f"semantic_generator_epoch_{epoch}.pth")
        torch.save(master_critic.state_dict(), f"multiscale_critic_epoch_{epoch}.pth")



