from typing import List, Dict, Optional
from torch.utils.data import Dataset
import albumentations as albu


class InMemoryDataset(Dataset):
    def __init__(self, data: List[Dict], transform: Optional[albu.Compose]=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.transform(**self.data[item]) if self.transform else self.data[item]
