import torch
import numpy as np
from torch.utils.data import Dataset
from mmengine.registry import DATASETS
import scipy.io as sio

@DATASETS.register_module()
class TrainDataset(Dataset):
    def __init__(self, data_root, length):
        Training_data= sio.loadmat(data_root)
        Training_labels =Training_data['matrices']
        self.data = Training_labels
        self.len = length
    def __getitem__ (self,index):
        #取出第 index 行的数据，并转换为 tensor
        return {'matrices': torch.Tensor(self.data[index, :]).float()}
    def __len__(self):
        return self.len