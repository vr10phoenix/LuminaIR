import os
import csv
import time
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import torch.optim as optim
from IR_SuperResolution.data_loader import MasterMultiCityDataset


RESTORMER_WEIGHTS = "/content/drive/MyDrive/Projects/LuminaIR/restormer_up.pth"   # change in Colab
OUTPUT_DIR = "/content/drive/MyDrive/Projects/LuminaIR/"
EPOCHS = 50
LR = 2e-4
WEIGHT_DECAY = 1e-4
EDGE_WEIGHT = 0.2
ACCUM_STEPS = 4
PATIENCE = 10
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

root_output_path = "/content/drive/MyDrive/Projects/LuminaIR/output_/patches"
master_dataset = MasterMultiCityDataset(root_dir=root_output_path)

train_size = int(0.8 * len(master_dataset))
val_size = len(master_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    master_dataset,
    [train_size, val_size],
    generator=generator
)

print(f"Total Images      : {len(master_dataset)}")
print(f"Training Samples  : {len(train_dataset)}")
print(f"Validation Samples: {len(val_dataset)}")

#Data_loaders
train_load = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

val_load = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
)

class CharbonnierLoss(nn.Module):
    """
    Mask-Aware Charbonnier Loss
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, mask=None):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()


class EdgeLoss(nn.Module):
    """
    Mask-Aware Edge Preservation Loss
    """
    def __init__(self):
        super().__init__()
        laplacian_kernel = torch.tensor(
            [[0., -1., 0.],
             [-1., 4., -1.],
             [0., -1., 0.]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("kernel", laplacian_kernel)
        self.charbonnier = CharbonnierLoss()

    def laplacian(self, x):
        channels = x.shape[1]
        kernel = self.kernel.repeat(channels, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=channels)

    def forward(self, pred, target, mask=None):
        pred_edges = self.laplacian(pred)
        target_edges = self.laplacian(target)

        # Pass the mask
        return self.charbonnier(pred_edges, target_edges, mask)


class SuperResolutionLoss(nn.Module):
    """
    Geospatial-Aware Super Resolution Loss
    Actively ignores padding, rotation artifacts, and NoData zones.
    """
    def __init__(self, edge_weight=0.2):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        with autocast(device_type="cuda", enabled=False):
            pred = pred.float()
            target = target.float()
            mask = (target > 0.0).float()
            pixel = self.pixel_loss(pred, target, mask=mask)
            edge = self.edge_loss(pred, target, mask=mask)
            total = pixel + self.edge_weight * edge

        return total


if __name__ == "__main__":

    criterion = SuperResolutionLoss()

    pred = torch.randn(2, 1, 128, 128)
    target = torch.randn(2, 1, 128, 128)
    loss = criterion(pred, target)
    print(f"Loss : {loss.item():.6f}")
    print("[SUCCESS] Super Resolution Loss Initialized.")


# torch.autograd.set_detect_anomaly(True)

train_loader = train_load
val_loader = val_load


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
torch.backends.cudnn.benchmark = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
restormer = Restormer().to(DEVICE)

state = torch.load(RESTORMER_WEIGHTS, map_location=DEVICE)
restormer.load_state_dict(state)

restormer.eval()

for p in restormer.parameters():
    p.requires_grad = False

swinir = SwinIR_Thermal(upscale=2).to(DEVICE)

criterion = SuperResolutionLoss(edge_weight=EDGE_WEIGHT).to(DEVICE)

optimizer = optim.AdamW(
    swinir.parameters(),
    lr=LR,
    betas=(0.9, 0.999),
    weight_decay=WEIGHT_DECAY,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)

scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
best_val = float("inf")
patience_counter = 0
csv_path = os.path.join(OUTPUT_DIR, "training_log.csv")

if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", "lr", "time_sec"]
        )

for epoch in range(EPOCHS):
    start = time.time()
    swinir.train()
    optimizer.zero_grad(set_to_none=True)
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, (lr_img, hr_img) in enumerate(pbar):

        lr_img = lr_img.to(DEVICE, non_blocking=True)
        hr_img = hr_img.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            clean = restormer(lr_img).detach()

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):

            pred = swinir(clean)
            loss = criterion(pred, hr_img)
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        if ((step + 1) % ACCUM_STEPS == 0) or ((step + 1) == len(train_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                swinir.parameters(),
                1.0,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item() * ACCUM_STEPS

        pbar.set_postfix(
            loss=f"{train_loss/(step+1):.5f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    train_loss /= len(train_loader)
    swinir.eval()

    val_loss = 0.0

    with torch.no_grad():

        for lr_img, hr_img in val_loader:

            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)

            clean = restormer(lr_img)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                pred = swinir(clean)
                loss = criterion(pred, hr_img)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scale = scaler.get_scale()
    if scale >= scaler.get_scale():
      scheduler.step()

    elapsed = time.time() - start

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f} | "
        f"Time {elapsed:.1f}s"
    )

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow( [epoch + 1,train_loss,val_loss,
                optimizer.param_groups[0]["lr"],elapsed,]
        )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": swinir.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
    }

    torch.save(checkpoint,os.path.join( OUTPUT_DIR,"last_checkpoint.pth",),
    )

    if val_loss < best_val:
        best_val = val_loss
        patience_counter = 0

        torch.save(
            swinir.state_dict(),
            os.path.join(
                OUTPUT_DIR,
                "best_swinir.pth",
            ),
        )

        print(" model updated.")

    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break

print("Training Complete.")
