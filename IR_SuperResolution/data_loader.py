import os
import numpy as np
import torch
from torch.utils.data import Dataset , DataLoader

class Landsat(Dataset):
  def __init__(self , root_dir):
    self.root_dir = root_dir
    self.sample_folders = []

    for city_name in sorted(os.listdir(root_dir)):
      city_path = os.path.join(root_dir , city_name)
      if os.path.isdir(city_path):
        for folder_name in sorted(os.listdir(city_path)):
          sample_path = os.path.join(city_path , folder_name)

          if os.path.isdir(sample_path) and folder_name.startswith("sample_"):
            self.sample_folders.append(sample_path)
            print(sample_path)


    print(f" Discovered {len(self.sample_folders)} sample folders in the directory")

root_data_path = '/content/drive/MyDrive/Projects/LuminaIR/output/patches'
dataset = Landsat(root_data_path)


class MasterMultiCityDataset(Dataset):
    """ Landsat Dataset Loader """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sample_paths = []

        # to see through given input structure of files
        for city_name in sorted(os.listdir(root_dir)):
            city_path = os.path.join(root_dir, city_name)

            if os.path.isdir(city_path):
                for sample_name in sorted(os.listdir(city_path)):
                    sample_path = os.path.join(city_path, sample_name)

                    if os.path.isdir(sample_path) and sample_name.startswith('sample_'):
                        self.sample_paths.append(sample_path)
        
        # checks to confirm samples
        print(f"Scanned root directory successfully.")
        print(f" sample folders locked: {len(self.sample_paths)}")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_dir = self.sample_paths[idx]

        tir_200m_path = os.path.join(sample_dir, 'tir_200m.npy')
        tir_100m_path = os.path.join(sample_dir, 'tir_100m_512.npy')

        tir_200m_np = np.load(tir_200m_path)
        tir_100m_np = np.load(tir_100m_path)

        # convert  PyTorch tensors
        tir_200m = torch.from_numpy(tir_200m_np).float()
        tir_100m = torch.from_numpy(tir_100m_np).float()

        if tir_200m.dim() == 2:
            tir_200m = tir_200m.unsqueeze(0)
        elif tir_200m.dim() == 3 and tir_200m.shape[-1] == 1:
            tir_200m = tir_200m.permute(2, 0, 1)

        # Normalize the 100m ground truth tensor
        if tir_100m.dim() == 2:
            tir_100m = tir_100m.unsqueeze(0)
        elif tir_100m.dim() == 3 and tir_100m.shape[-1] == 1:
            tir_100m = tir_100m.permute(2, 0, 1)

        return tir_200m, tir_100m


# master 'output' directory in Drive
root_output_path = '/content/drive/MyDrive/Projects/LuminaIR/output/patches'

train_dataset = MasterMultiCityDataset(root_dir=root_output_path)

# batches = 2 to run in colab T4 GPU
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
