import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
import numpy as np
from typing import Dict, Iterable, NewType, Sequence, Tuple, Optional


class LocalImageDataset(Dataset):
    def __init__(self, root_path:str, image_limit:Optional[int]=None) -> None:
        self.root_path = root_path
        self.source_files = []
        for dir_path, _, file_names in os.walk(self.root_path):
            for file_name in file_names:
                full_path = os.path.join(dir_path, file_name)
                self.source_files.append(full_path)
        if image_limit:
            self.source_files = self.source_files[:image_limit]

    def __len__(self) -> int:
        return len(self.source_files)

    def __getitem__(self, idx:int) -> Iterable[torch.Tensor]:
        im = Image.open(self.source_files[idx])
        return [ToTensor()(im)]
        #return tuple(result)
