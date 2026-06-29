import os 
import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataloader import MasterMultiCityDataset , DataLoader
from restormer import pre_restormer


root_output_path ='/kaggle/working/patches'

train_dataset = MasterMultiCityDataset(root_dir=root_output_path)

train_load = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

class CharbonnierLoss(nn.Module):
    """L1 Loss."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))

# Initialize the Loss
charbonnier_criterion = CharbonnierLoss().cuda()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights_url = "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth"
weights_path = "real_denoising.pth"

if not os.path.exists(weights_path):
    print("Downloading Official Pre-trained Restormer Weights (SIDD)...")
    !wget {weights_url} -O {weights_path}
else:
    print("Weights already downloaded!")

restormer = pre_restormer(pretrained_weights_path=weights_path, device="cuda")
restormer.to(device)
restormer.eval()

print(" Locking deep Restormer weights. Unlocking surgical layers...")

for name, param in restormer.named_parameters():

    if 'embed_conv' in name or 'mapping' in name:
        param.requires_grad = True
        print(f"  : [UNLOCKED] {name}") 
    else:
        param.requires_grad = False
optimizer_R = optim.AdamW(
    filter(lambda p: p.requires_grad, restormer.parameters()),
    lr=2e-4,
    weight_decay=1e-4
)
print("[SUCCESS] Optimizer loaded with surgical parameters. Ready for Training.")

restormer.train()
epochs_restormer = 15 

print("========== INITIATING PHASE 2A: RESTORMER WARM-UP ==========")
for epoch in range(epochs_restormer):
    epoch_loss = 0.0
    start_time = time.time()

    for i, (raw_ir, _) in enumerate(train_load):
        raw_ir = raw_ir.cuda()

        synthetic_noise = torch.randn_like(raw_ir) * 0.05
        noisy_ir = raw_ir + synthetic_noise

        optimizer_R.zero_grad()
        cleaned_ir = restormer(noisy_ir)
        loss = charbonnier_criterion(cleaned_ir, raw_ir)
        loss.backward()
        optimizer_R.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_load)
    print(f"Epoch [{epoch+1}/{epochs_restormer}] | Time: {time.time() - start_time:.2f}s | Charbonnier Loss: {avg_loss:.6f}")

# Save weights
save_dir = '/kaggle/working/'
save_filename = 'restormer_up.pth'
save_path = os.path.join(save_dir, save_filename)

# Save the model
torch.save(restormer.state_dict(), save_path)
if os.path.exists(save_path):
    print(f"[SUCCESS] Model successfully saved to: {save_path}")
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
else:
    print("[ERROR] Failed to save the model.")
print("[SUCCESS] Restormer adapted to Thermal Data and saved.")