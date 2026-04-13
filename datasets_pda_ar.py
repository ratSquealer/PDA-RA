import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from curriculum import brightness_score_pil


class PairedAugImageFolder(Dataset):
    def __init__(self, root, transform_source=None, transform_target=None):
        self.ds = ImageFolder(root=root, transform=None)
        self.transform_source = transform_source
        self.transform_target = transform_target
        self.sorted_indices = self._build_sorted_indices()

    def _build_sorted_indices(self):
        scores = []
        for idx, (path, _) in enumerate(self.ds.samples):
            img = Image.open(path).convert("RGB")
            score = brightness_score_pil(img)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scores]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        path, label = self.ds.samples[idx]
        img = Image.open(path).convert("RGB")
        img_s = self.transform_source(img) if self.transform_source else img
        img_t = self.transform_target(img) if self.transform_target else img_s
        return img_s, img_t, label

