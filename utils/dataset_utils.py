import os

import numpy as np
import scipy.io as sio
import scipy.ndimage as nd
import torch
import torch.utils.data as data

class ShapeNet(data.Dataset):
    def __init__(
            self,
            root: str,
            category: str,
            voxel_resolution: int=64,
            image_set: str="train",
        ) -> None:
        super().__init__()
        self.root = root
        self.image_set = image_set
        self.voxel_resolution = voxel_resolution
        self.category = category

        volumetric_dir = os.path.join(self.root, self.category, "30", self.image_set)
        self.file_names = [
            os.path.join(volumetric_dir, x).strip()
            for x in os.listdir(volumetric_dir) if x.endswith(".mat")
        ]

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        voxel = sio.loadmat(self.file_names[idx])["instance"]
        voxel = np.pad(voxel, pad_width=(1, 1), mode="constant", constant_values=(0, 0))
        if self.voxel_resolution != 32:
            ratio = self.voxel_resolution / 32.
            voxel = nd.zoom(voxel, zoom=(ratio, ratio, ratio), mode="constant", order=0)
        voxel = np.expand_dims(voxel.astype(np.float32), axis=0)
        return torch.from_numpy(voxel)
