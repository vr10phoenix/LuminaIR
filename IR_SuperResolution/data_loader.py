import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MasterMultiCityDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.sample_paths = []
        
        # Load all sample paths
        for city_name in sorted(os.listdir(root_dir)):
            city_path = os.path.join(root_dir, city_name)
            if os.path.isdir(city_path):
                for sample_name in sorted(os.listdir(city_path)):
                    sample_path = os.path.join(city_path, sample_name)
                    if os.path.isdir(sample_path) and sample_name.startswith('sample_'):
                        self.sample_paths.append(sample_path)
        
        # Sort them so we get consistent splits
        self.sample_paths.sort()
        
        # ---------- TRAIN/VAL SPLIT (80/20) ----------
        # Extract the sample number from the path (e.g., sample_120)
        train_samples = [p for p in self.sample_paths if int(p.split('sample_')[-1]) <= 100]
        val_samples = [p for p in self.sample_paths if int(p.split('sample_')[-1]) > 100]
        
        if train:
            self.sample_paths = train_samples
            print(f"Training Set: {len(self.sample_paths)} samples")
        else:
            self.sample_paths = val_samples
            print(f"Validation Set: {len(self.sample_paths)} samples")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]
        lr_path = os.path.join(sample_dir, 'tir_200m.npy')
        hr_path = os.path.join(sample_dir, 'tir_100m_512.npy')
        lr_np = np.load(lr_path)
        hr_np = np.load(hr_path)
        lr = torch.from_numpy(lr_np).float()
        hr = torch.from_numpy(hr_np).float()
        if lr.dim() == 2:
            lr = lr.unsqueeze(0)
        if hr.dim() == 2:
            hr = hr.unsqueeze(0)
        return lr, hr

ROOT_PATH = r'D:\Project\ir\LuminaIR\Pipeline\Preprocessing\patches'

# Create training and validation loaders
train_dataset = MasterMultiCityDataset(root_dir=ROOT_PATH, train=True)
val_dataset = MasterMultiCityDataset(root_dir=ROOT_PATH, train=False)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)