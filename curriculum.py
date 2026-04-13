import math
import random

import torch
from torch.utils.data import Sampler


def brightness_score_pil(img):
    if img.mode != "L":
        img = img.convert("L")
    t = torch.from_numpy(__import__("numpy").array(img, dtype="float32"))
    return float(t.sum().item())


class LearningSortingSampler(Sampler):
    def __init__(
        self,
        sorted_indices,
        dataset_size,
        start_ratio=0.4,
        increment=0.15,
        step_epochs=10,
        seed=3407,
    ):
        self.sorted_indices = list(sorted_indices)
        self.dataset_size = int(dataset_size)
        self.start_ratio = float(start_ratio)
        self.increment = float(increment)
        self.step_epochs = int(step_epochs)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _current_ratio(self):
        steps = self.epoch // self.step_epochs
        return min(1.0, self.start_ratio + steps * self.increment)

    def __iter__(self):
        ratio = self._current_ratio()
        n = max(1, int(self.dataset_size * ratio))
        idx = self.sorted_indices[:n]
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(idx)
        return iter(idx)

    def __len__(self):
        ratio = self._current_ratio()
        return max(1, int(self.dataset_size * ratio))

