import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset




class CubDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(os.path.join(root_dir, csv_path))
        self.samples = df

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'].replace('', 'jpg/', 1))
        image = Image.open(img_path).convert('RGB')
        label = int(row['class_idx'])

        if self.transform:
            image = self.transform(image)

        return image, label
