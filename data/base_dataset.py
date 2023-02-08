

from torch.utils.data import Dataset
from torchvision import datasets
import torch
import numpy as np

# ref : https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


class BaseDataSet(Dataset):
    def __init__(self):
        self.data = np.array([1,2,3,4,5])
        self.label = np.array([1,0,1,0,1])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data_ = torch.Tensor(self.data[idx]).cuda()
        label_ = torch.Tensor(self.label[idx]).cuda()
        return data_,label_

