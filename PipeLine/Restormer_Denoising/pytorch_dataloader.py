import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

class LandsatDataset(Dataset):
    def __init__(self, patches_dir):
        """
        Scans the training_patches folder and perfectly pairs the IR and RGB files.
        """
        self.patches_dir = Path(patches_dir)
        
        self.ir_files = sorted(list(self.patches_dir.glob("ir_patch_*.npy")))
        self.rgb_files = sorted(list(self.patches_dir.glob("rgb_patch_*.npy")))
        
        # Quick sanity check
        assert len(self.ir_files) == len(self.rgb_files), "CRITICAL: Mismatch in number of IR and RGB patches!"
        assert len(self.ir_files) > 0, "No patches found. Check your directory path!"

    def __len__(self):
        # no. of pairs
        return len(self.ir_files)

    def __getitem__(self, idx):
        """
        This function runs on the fly during training. It loads a single pair, 
        formats it, and converts it to a PyTorch Tensor.
        """
        #load the NumPy arrays
        ir_patch = np.load(self.ir_files[idx])
        rgb_patch = np.load(self.rgb_files[idx])

        # Channel Formatting 
        ir_patch = np.expand_dims(ir_patch, axis=0) # Becomes (1, 256, 256)

        #Tensor Conversion
        ir_tensor = torch.from_numpy(ir_patch).float()
        rgb_tensor = torch.from_numpy(rgb_patch).float()

        return ir_tensor, rgb_tensor

def get_dataloaders(patches_dir, batch_size=16, train_split=0.85):
    """
    Splits the dataset and creates the DataLoaders for batching.
    """
    dataset = LandsatDataset(patches_dir)
    
    # Calculate exactly how many images go into Training vs Validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Randomly shuffle and split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader

patches_directory = "training_patches"
train_loader, val_loader = get_dataloaders(patches_directory, batch_size=16)