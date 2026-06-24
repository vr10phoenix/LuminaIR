import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from data_loader import train_loader
from restormer import pre_restormer
from swinir import SwinIR_Thermal
from loss_function import SuperResolutionLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights_url = "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"
weights_path = "real_denoising.pth"

if not os.path.exists(weights_path):
    print("Downloading Official Pre-trained Restormer Weights (SIDD)...")
    !wget {weights_url} -O {weights_path}
else:
    print("Weights already downloaded!")

# calling the models
restormer = pre_restormer(pretrained_weights_path=weights_path, device="cuda")
restormer.to(device)
restormer.eval()

# freeze upstream parameters (to save memory)
for param in restormer.parameters():
    param.requires_grad = False

swinir = SwinIR_Thermal(upscale=2).to(device)
swinir.train()

# optimizer
optimizer = optim.AdamW(swinir.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
criterion = SuperResolutionLoss(edge_weight=0.5).to(device)

epochs = 50
accumulation_steps = 4 

print("========== STARTING FOOTPRINT-OPTIMIZED RESTORATION TRAINING ============")

for epoch in range(epochs):
    epoch_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    for i, (tir_200m, tir_100m) in enumerate(train_loader):
        tir_200m = tir_200m.to(device)
        tir_100m = tir_100m.to(device)

        with torch.no_grad():
            clean_lr_thermal = restormer(tir_200m)

        sr_thermal = swinir(clean_lr_thermal)

        # Scale loss relative to gradient accumulation intervals
        loss = criterion(sr_thermal, tir_100m) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(swinir.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps

    scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1:02d}/{epochs}] | Compound Loss: {avg_loss:.6f} | Sec/Epoch: {time.time() - start_time:.1f}s")

# saving the weights file
torch.save(swinir.state_dict(), 'swinir_thermal_v1.pth')
print("Trained parameters cleanly written to 'swinir_thermal_v1.pth'.")