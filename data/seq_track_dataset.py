

from torch.utils.data import Dataset
from torchvision import datasets
from torch.nn.utils.rnn import PackedSequence

import torch
import numpy as np

# ref : https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#
# class TrackDataSet(PackedSequence,Dataset):
#     def __init__(self, track_len=120, wh=(128, 256), **kwargs):
#         super().__init__()
#         self.track_len = track_len
#         self.width = wh[0]
#         self.height = wh[1]
#
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

