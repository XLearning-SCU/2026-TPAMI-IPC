import json
from PIL import Image
import os
from typing import Callable
from typing import Any
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm


class Clevr4(Dataset):
    all_taxonomies = {
        'color': ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow", "pink", "orange"],
        'texture': ["rubber", "metal", "checkered", "emojis", "wave", "brick", "star", "circles", "zigzag",
                    "chessboard"],
        'count': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'shape': ["cube", "sphere", "monkey", "cone", "torus", "star", "teapot", "diamond", "gear", "cylinder"]
    }

    def __init__(
            self,
            root: str,
            taxonomy: str,
            transform: Callable = None,
    ):
        super().__init__()

        # Clevr4 Metadata
        self.root = root
        self.image_root = os.path.join(root, 'images')
        self.taxonomy = taxonomy
        self.transform = transform

        # Load annotations
        annot_path = os.path.join(root, 'clevr_4_annots.json')
        with open(annot_path, 'r') as f:
            annots = json.load(f)
        self.annotations = annots
        self.class_name_to_label = {
            name: idx for idx, name in enumerate(self.all_taxonomies[taxonomy])
        }

        # List files
        self.filenames = sorted(
            [fname for fname, meta in annots.items()]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        # Load image
        fname = self.filenames[index]
        img_path = os.path.join(self.image_root, f"{fname}.png")
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # Label to index
        class_name = self.annotations[fname][self.taxonomy]
        target = self.class_name_to_label[class_name]

        return image, target

    def _get_all_taxonomy_targets(self, taxonomy):

        class_name_to_label = {
            name: idx for idx, name in enumerate(self.all_taxonomies[taxonomy])
        }

        all_targets = []
        for fname in self.filenames:
            class_name = self.annotations[fname][taxonomy]
            target = class_name_to_label[class_name]
            all_targets.append(target)

        return all_targets




