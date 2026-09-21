import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset




class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, csv_path, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        df = pd.read_csv(os.path.join(root_dir, csv_path))

        # 根据是否为训练集进行筛选
        split_prefix = 'train/' if train else 'test/'
        self.samples = df[df['filename'].str.startswith(split_prefix)].reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'].replace('train/', 'cars_train/').replace('test/', 'cars_test/'))
        image = Image.open(img_path).convert('RGB')
        label = int(row['class_idx'])

        if self.transform:
            image = self.transform(image)

        return image, label
