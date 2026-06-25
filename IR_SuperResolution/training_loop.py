import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim
import time
import wget
from data_loader import train_loader, val_loader
from restormer import pre_restormer
from swinir import SwinIR_Thermal
from loss_function import SuperResolutionLoss

torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

weights_path = "real_denoising.pth"
if not os.path.exists(weights_path):
    print("Downloading pre-trained Restormer weights...")
    wget.download("https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth", weights_path)
else:
    print("Weights already downloaded.")

restormer = pre_restormer(pretrained_weights_path=weights_path, device=device)
restormer.to(device)
restormer.eval()
for p in restormer.parameters():
    p.requires_grad = False

swinir = SwinIR_Thermal(upscale=2, dim=32, depths=[4, 4, 4, 4], num_heads=8).to(device)
swinir.train()

optimizer = optim.AdamW(swinir.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
criterion = SuperResolutionLoss(edge_weight=0.5).to(device)

epochs = 30
accum_steps = 4
scaler = torch.amp.GradScaler('cuda')

print("========== TRAINING STARTED (TRAIN/VAL SPLIT ENABLED) ==========")

for epoch in range(epochs):
    # --- Training Phase ---
    epoch_loss = 0.0
    start = time.time()
    optimizer.zero_grad()
    
    total_batches = len(train_loader)
    for i, (lr, hr) in enumerate(train_loader):
        lr = lr.to(device)
        hr = hr.to(device)

        with torch.no_grad():
            clean_lr = restormer(lr)

        with torch.amp.autocast('cuda'):
            sr = swinir(clean_lr)
            loss = criterion(sr, hr) / accum_steps

        scaler.scale(loss).backward()

        if (i+1) % accum_steps == 0 or (i+1) == total_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(swinir.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accum_steps
        
        if (i+1) % 10 == 0:
            print(f"  Batch {i+1}/{total_batches} | Loss: {loss.item() * accum_steps:.6f}")

    avg_train_loss = epoch_loss / total_batches

    # --- Validation Phase (Unseen Data) ---
    val_loss = 0.0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            clean_lr = restormer(lr)
            sr = swinir(clean_lr)
            val_loss += criterion(sr, hr).item()
    
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step()
    
    print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {time.time()-start:.1f}s")

torch.save(swinir.state_dict(), 'swinir_thermal_v1.pth')
print("Model saved as swinir_thermal_v1.pth")