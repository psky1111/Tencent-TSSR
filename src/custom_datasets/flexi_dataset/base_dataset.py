import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, dataset_args, project_args, debug_mode=False, mode="train", batch_size=2, dtype=torch.float16, 
                 device=None, **kwargs):
        self.dataset_args = dataset_args
        self.project_args = project_args
        self.debug_mode = debug_mode
        self.mode = mode
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
