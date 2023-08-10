import torch
import numpy as np


class MNISTCollator:
    def __call__(self, batch):
        x = torch.tensor(
            np.array([np.asarray(b["image"]).flatten() / 255 for b in batch])
        ).to(torch.float32)
        y = torch.tensor([b["label"] for b in batch]).long()
        y_onehot = torch.nn.functional.one_hot(y, num_classes=10).to(torch.float32)
        return {
            "image": x,
            "target": y,
            "target_onehot": y_onehot,
        }
