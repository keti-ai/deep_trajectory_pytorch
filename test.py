
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.post_net import SiameseBiGRU, Cleaving

import numpy as np

# Logging
import visdom
import datetime
# Text
class VisLogger():
    def __init__(self,title="Test",env_="main",legends=["1","2"]):
        self.vis = visdom.Visdom(env=env_)
        self.vis.close(env=env_)
        # EXP = "Train Cleaving\n"
        # self.vis.text(EXP + "<br>" + str(datetime.datetime.now()), env="main")
        self.data_len=0
        self.wins=[]
        self._x=0
        self.title=title
        self.legends=legends
    def vis_log(self,win_name,x_, val,title="Test"):

        # val=torch.view()
        try:
            self.data_len =len(val)
        except:
            self.data_len=0
        if self.data_len > 1:
            val = [li.item() for li in val]
            self.vis.line(Y=[val],X=[x_],win=win_name, update="append" ,opts=dict(title=title, legend=self.legends,showlegend=True))
        else:
            val=val.item()
            self.vis.line(Y=[val], X=[x_], win=win_name, update="append",opts=dict(title=title))

        self.wins.append(win_name)
        self.wins=list(set(self.wins))
        self._x=x_
    def vis_img(self,win_name,img):
        if len(img.shape)>3:
            self.vis.images(img,win=win_name)
        else:
            self.vis.image(img,win=win_name)


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

from torch.utils.data import DataLoader
from tqdm import tqdm
from data.seq_track_dataset import BaseDataSet,TrackDataSet
from torch.utils.data.sampler import BatchSampler, SequentialSampler,SubsetRandomSampler,RandomSampler
import torch.optim as optim
import time
class BaseDataLoader(DataLoader):
    def __init__(self, dataset=BaseDataSet, batch_size=1, shuffle=False, sampler=None,batch_sampler=None, num_workers=0, collate_fn=None,pin_memory=False, drop_last=False, timeout=0,worker_init_fn=None, *, prefetch_factor=2,persistent_workers=False):
        super().__init__(dataset=dataset,persistent_workers=persistent_workers,batch_size=batch_size,shuffle=shuffle,sampler=sampler,drop_last=drop_last)
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
    batch_size = samples.__len__()
    for i in range(batch_size):
        in_.append(samples[i][0])
        label_.append(samples[i][1])
    return samples#{"input":in_,"label":label_}

# Train the model

import os
if __name__ == "__main__":

    vis_log=VisLogger(title="Test_loss",env_="test",legends=["feat_loss","srh_loss","pur_loss","loss"])
    epochs = 100

    feat_size = 2048  # size of input features
    hidden_size = 128
    num_layers = 4
    dropout = 0.2
    learning_rate = 0.001

    # Tracklet info
    track_len_in = 120

    # Create the model
    # model = Cleaving(track_len=track_len_in, id_len=22, feat_size=feat_size, hidden_size=hidden_size,
    #                  num_layers=num_layers, dropout=dropout).cuda()

    # Define the loss function and optimizer
    trackdataset = TrackDataSet(root_dir="/media/syh/ssd2/data/ReID/bounding_box_train",track_length=120)
    batch_size_ = 1
    dataloader = BaseDataLoader(
        dataset=trackdataset,
        collate_fn=base_collate_fn,
        batch_size=batch_size_,
        sampler=BatchSampler(
            RandomSampler(trackdataset),
            batch_size=batch_size_,
            drop_last=False
        )
    )
    model=torch.load("/media/syh/hdd/checkpoints/deep_trajectory/1488_model.pth")
    model.eval()

    for i,data in enumerate(dataloader):
        label = data[0][1]
        tracklet = data[0][0]
        gru_out_f, gru_out_b, srh_out, pur_out,cls_out = model(torch.stack([tracklet]))
        y_pred_f=gru_out_f.topk(5)
    # tracklets=next(iter(dataloader))
    st=time.time()
    gru_out_f, gru_out_b, srh_out, pur_out = model(torch.stack([tracklets[0][0]]))
    ed=time.time()
    print("infer time : ",ed-st)