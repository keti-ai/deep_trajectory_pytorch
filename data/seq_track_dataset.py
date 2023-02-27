

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
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
        self.data = np.random.rand(10,120,256,128,3)
        self.label = np.random.randint(0,1000,(10,120))
        self.transforms=transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((224,224))])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): # idx == iter
        in_batch=self.data[idx].__len__()
        data_=[]
        for j in range(in_batch):
            data_in=self.data[idx][j]
            for i in range(data_in.shape[0]):
                data_.append(self.transforms(data_in[i]))
        return torch.Tensor(torch.stack(data_)).float().cuda(),torch.Tensor(self.label[idx]).cuda().type(torch.cuda.LongTensor)

