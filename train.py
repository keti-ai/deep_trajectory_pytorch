
import torch
import torch.nn as nn
import torch.nn.functional as F

import visdom
vis = visdom.Visdom()

EXP="Train Cleaving\n"

# Text
import datetime
vis.text(EXP+"<br>"+str(datetime.datetime.now()),env="main")
from model.seresnext import seresnext50_32x4d

from torch.nn.utils.rnn import PackedSequence
class Cleaving(nn.Module):
    def __init__(self,wh=(128,258),track_len=120,id_len=1000,feat_size=2048, hidden_size=256, num_layers=4, dropout=0.2):

        # nn model
        self.feat_ex_module=seresnext50_32x4d(in_size=wh)
        self.siamModel = SiameseBiGRU(feat_size, hidden_size, num_layers, dropout)

        self.feat_fcn = nn.Linear(track_len, id_len)
        self.srh_fcn = nn.Linear(track_len - 1, track_len - 1)
        self.pur_fcn = nn.Linear(track_len - 1, 2)

        self.width = wh[0]
        self.height = wh[1]

        # loss

        self.criterion_feat
        self.cr
    def forward(self,x):

        feat=self.feat_ex_module(x)
        print(feat.shape)
        feat = PackedSequence(feat)
        phi_g_for,phi_g_back=self.siamModel(feat,feat[::-1]) # output.view(seq_len, batch, num_directions, hidden_size) output.view(seq_len, batch, num_directions, hidden_size)
        gru_output_f = self.feat_fcn(phi_g_for) # seq_len batch id_len
        gru_output_b = self.feat_fcn(phi_g_back) # seq_len batch id_len


        phi_d = (phi_g_for[:-1]-phi_g_back[:-1][::-1]).pow(2).sum(3).sqrt()

        srh_output = self.srh_fcn(phi_d)
        pur_output = self.pur_fcn(phi_d)

        return gru_output_f,gru_output_b,srh_output,pur_output



class SiameseBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(SiameseBiGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward_once(self, x, _forward ):
        # x.shape = L, batch, h_in
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)  # *2 for bidirection
        output, _ = self.gru(x,h0)
        # out = torch.mean(output, dim=0)
        if _forward :
            return output.view(-1, -1, 0, self.hidden_size)
        if not _forward :
            return output.view(-1, -1, 1, self.hidden_size)
    def forward(self, x):
        # x.shape = L, batch, h_in

        # out = torch.mean(output, dim=0)
        out_forward = forward_once(x,_forward=True)
        out_backward = forward_once(x[::-1],_forward=False)
        return out_forward,out_backward


criterion = nn.BCEWithLogitsLoss()


'''
                 channels,
                 init_block_channels,
                 cardinality,
                 bottleneck_width,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
'''
# Define the hyperparameters
feat_size = 2048 # size of input features
hidden_size = 256
num_layers = 4
dropout = 0.2
learning_rate = 0.001

# Tracklet info
track_len_in = 120

# Create the model
model = Cleaving(track_len=track_len_in,id_len=1000,feat_size, hidden_size, num_layers, dropout)

# Define the loss function and optimizer
lam_feat = 1
lam_search = 1
lam_pur = 1

criterion_gru = torch.nn.CrossEntropyLoss()
criterion_srh = torch.nn.CrossEntropyLoss()
criterion_pur = torch.nn.CrossEntropyLoss()

cleav_loss = lam_feat*feat_loss + lam_search*search_loss + lam_pur * pur_loss

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from torch.utils.data import DataLoader
from tqdm import tqdm
from data.base_dataset import BaseDataSet
from torch.utils.data.sampler import BatchSampler, SequentialSampler

import time
class BaseDataLoader(DataLoader):
    def __init__(self, dataset=BaseDataSet, batch_size=1, shuffle=False, sampler=None,batch_sampler=None, num_workers=0, collate_fn=None,pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None, *, prefetch_factor=2,persistent_workers=False):
        super().__init__(dataset=dataset,persistent_workers=persistent_workers,batch_size=batch_size,sampler=sampler,drop_last=drop_last)
        # self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.worker_init_fn = worker_init_fn
        self.timeout = timeout
        self.collate_fn = collate_fn
        # self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        # self.batch_sampler = batch_sampler
        # self.sampler = samplers
        self.shuffle = shuffle
        # self.batch_size = batch_size
def base_collate_fn(samples):
    in_=[]
    label_=[]
    for i,j in samples:
        in_.append(i)
        label_.append(j)
    return {"input":in_,"label":label_}

basedataset = BaseDataSet()
dataloader = BaseDataLoader(
    dataset=basedataset,
    collate_fn=base_collate_fn,
    batch_size=1,
    sampler=BatchSampler(
        SequentialSampler(basedataset),
        batch_size=1,
        drop_last=False
    )
)
# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (tracklet, label) in enumerate(dataloader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        gru_out_f,gru_out_b,srh_out,pur_out = model(tracklet)


        loss = criterion(output1 - output2, label.float())


        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[Epoch %d, Iteration %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
