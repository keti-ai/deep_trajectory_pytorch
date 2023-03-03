
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def vis_log(self,win_name,x_, val):

        # val=torch.view()
        self.data_len =len(val)
        if self.data_len > 1:
            val = [li.item() for li in val]
            self.vis.line(Y=[val],X=[x_],win=win_name, update="append" ,opts=dict(title=self.title, legend=self.legends,showlegend=True))
        else:
            self.vis.line(Y=[val], X=[x_], win=win_name, update="append")

        self.wins.append(win_name)
        self.wins=list(set(self.wins))
        self._x=x_
    def vis_img(self,win_name,img):
        if len(img.shape)>3:
            self.vis.images(img,win=win_name)
        else:
            self.vis.image(img,win=win_name)



from model.seresnext import seresnext50_32x4d
# from torch.nn.utils.rnn import PackedSequence
class Cleaving(nn.Module):
    def __init__(self,wh=(224,224),track_len=120,id_len=1000,feat_size=2048, hidden_size=256, num_layers=4, dropout=0.2):
        super().__init__()
        # params
        self.track_len=track_len
        self.wh = wh

        # nn model
        self.feat_ex_module=seresnext50_32x4d(in_size=wh)

        self.siamModel = SiameseBiGRU(track_len,feat_size, hidden_size, num_layers, dropout).cuda()
        self.feat_fcn = nn.Linear(256, id_len)

        # self.srh_fcn = nn.Linear(track_len - 1, track_len - 1)
        self.srh_fcn = nn.Sequential(
            nn.Linear(track_len - 1, track_len - 1),
            nn.Sigmoid()
        )
        self.pur_fcn = nn.Sequential(
            nn.Linear(track_len - 1, 2),
            nn.Linear(2,1),
            nn.Sigmoid()
        )

        self.width = wh[0]
        self.height = wh[1]

    def forward(self,x): # x.shape == batch,track, feat
        batch_in=x.shape[0]
        x=x.view(self.track_len,batch_in,-1,self.wh[0],self.wh[1])
        feat = []
        for i in range(self.track_len):
            feat.append(self.feat_ex_module(x[i]))
        # feat=self.feat_ex_module(x)

        feat = torch.stack(feat)
        feat=feat.view(batch_in,self.track_len,-1)
        # print("batch_in : ",torch.Tensor(batch_in))
        # feat_for = PackedSequence(feat,batch_sizes=torch.Tensor([1,120]))
        # print("len(feat_for) : ",len(feat_for))
        feat_back = torch.flip(feat,[0])
        # feat_back = PackedSequence(feat_back,batch_sizes=torch.Tensor([1,120]))


        # siamnes GRU module
        phi_g_for,phi_g_back=self.siamModel(feat,feat_back) # output.view(seq_len, batch, num_directions, hidden_size) output.view(seq_len, batch, num_directions, hidden_size)

        gru_output_f = self.feat_fcn(phi_g_for) # seq_len batch id_len
        gru_output_b = self.feat_fcn(phi_g_back) # seq_len batch id_len

        gru_output_f=gru_output_f
        gru_output_b=gru_output_b
        phi_d = (phi_g_for[:,1:]-torch.flip(phi_g_back[:,1:],[0])).pow(2).sum(2).sqrt()

        srh_output = self.srh_fcn(phi_d)
        pur_output = self.pur_fcn(phi_d)
        return gru_output_f,gru_output_b,srh_output,pur_output




class SiameseBiGRU(nn.Module):
    def __init__(self, track_len=120,input_size=2048, hidden_size=256, num_layers=4, dropout=0.2):
        super(SiameseBiGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.track_len=track_len
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward_once(self, x, h,_forward ):
        # x.shape = L, batch, h_in
        # h0 = torch.zeros(self.num_layers*2,len(x), self.hidden_size).cuda()  # *2 for bidirection
        # print("h0 ",h0.shape)
        output, h_out = self.gru(x,h)

        # out = torch.mean(output, dim=0)
        if _forward :
            output=output[:,:,:self.hidden_size]
            return output.view(len(x), -1, self.hidden_size),h_out
        if not _forward :
            output = output[:,:,self.hidden_size:]
            return output.view(len(x),-1, self.hidden_size),h_out
    def forward(self, x,x_): # x shape == L*Hin
        # print("x.shape ",x.shape)
        # x.shape = L, batch, h_in
        out_forward=[]
        out_backward=[]
        # out = torch.mean(output, dim=0)
        # print("len(x) : ",len(x))
        hn_f = torch.zeros(self.num_layers * 2,  self.track_len,self.hidden_size).cuda()
        hn_b = torch.zeros(self.num_layers * 2,  self.track_len,self.hidden_size).cuda()
        # print("x[0].shape :",x[0].shape)

        out_f,hn_f=self.forward_once(x,hn_f,_forward=True)
        out_b,hn_b=self.forward_once(x_,hn_b,_forward=False)
        # for i in range(x.shape[0]):
        #     out_f,hn_f=self.forward_once(x[i],hn_f,_forward=True)
        #     out_forward.append(out_f)
        #     out_b,hn_b=self.forward_once(x_[i],hn_b,_forward=False)
        #     out_backward.append(out_b)
        return out_f,out_b

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
    batch_size = samples.__len__()
    for i in range(batch_size):
        in_.append(samples[i][0])
        label_.append(samples[i][1])
    return samples#{"input":in_,"label":label_}

# Train the model

import os
if __name__ == "__main__":

    vis_log=VisLogger(title="Test_loss",env_="main",legends=["feat_loss","srh_loss","pur_loss","loss"])
    num_epochs = 10

    feat_size = 2048  # size of input features
    hidden_size = 256
    num_layers = 4
    dropout = 0.2
    learning_rate = 0.1

    # Tracklet info
    track_len_in = 120

    # Create the model
    model = Cleaving(track_len=track_len_in, id_len=22, feat_size=feat_size, hidden_size=hidden_size,
                     num_layers=num_layers, dropout=dropout).cuda()

    # Define the loss function and optimizer
    lam_feat = 1
    lam_search = 5
    lam_pur = 1

    criterion_gru = torch.nn.CrossEntropyLoss()
    criterion_srh = torch.nn.CrossEntropyLoss()
    criterion_pur = torch.nn.BCELoss()

    # cleav_loss = lam_feat*criterion_gru + lam_search*criterion_srh + lam_pur * criterion_pur

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # basedataset = BaseDataSet()

    basedataset = TrackDataSet(root_dir="/media/syh/ssd2/data/ReID/bounding_box_train",track_length=120)
    batch_size_ = 1
    dataloader = BaseDataLoader(
        dataset=basedataset,
        collate_fn=base_collate_fn,
        batch_size=batch_size_,
        sampler=BatchSampler(
            SequentialSampler(basedataset),
            batch_size=1,
            drop_last=False
        )
    )

    iter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            iter+=1
            batch_size = data.__len__()
            tracklets = []
            labels = []
            for j in range(batch_size):
                tracklets.append(data[j][0])
                labels.append(data[j][1].squeeze())
            # print("tracklet.__len__():", tracklet.__len__())
            # Zero the parameter gradients
            optimizer.zero_grad()
            # print(tracklet.type)
            # Forward + backward + optimize
            gru_out_f, gru_out_b, srh_out, pur_out = model(torch.stack(tracklets))

            label = torch.stack(labels)

            # loss = criterion(output1 - output2, label.float())
            loss_ = 0
            loss_gru = 0
            loss_srh = 0
            loss_pur = 0


            for j in range(batch_size):
                # ToDo search network target 설정
                loss_gru += criterion_gru(gru_out_f[j], label[j]) + \
                            criterion_gru(gru_out_b[j], label[j])
                loss_srh += criterion_srh(srh_out[j], torch.Tensor([label[j][i] == label[j][i + 1] for i in range(len(label[j]) - 1)]).cuda())
                loss_pur += criterion_pur(pur_out[j], torch.Tensor([int(len(torch.unique(label))==1)]).cuda())

            loss_ = lam_feat * loss_gru + lam_search * loss_srh + lam_pur * loss_pur
            # vis_log.vis_log(win_name="main/loss",x_=iter,val=[loss_gru,loss_srh,loss_pur,loss_])
            loss_.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_.item()

            # print('[Epoch %d, Iteration %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            vis_log.vis_log(win_name="main/loss", x_=iter, val=[loss_gru, loss_srh, loss_pur, loss_])
            vis_log.vis_log(win_name="main/loss_Gru", x_=iter, val=loss_gru)
            vis_log.vis_log(win_name="main/loss", x_=iter, val=loss_srh)
            vis_log.vis_log(win_name="main/loss", x_=iter, val=loss_pur)
            if i % 100 == 99:
                vis_log.vis_img(win_name="main/input", img=data[0][0][0][::20])
                print('[Epoch %d, Iteration %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                PATH="%d_model.pth"%(iter)
                PATH=os.path.join("/media/syh/hdd/checkpoints/deep_trajectory",PATH)
                torch.save(model, PATH)