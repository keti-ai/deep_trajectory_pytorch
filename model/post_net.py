import torch
import torch.nn as nn
import torch.nn.functional as F

# from base_model import Base_Model

# bbox => 10 dim
# cid, id ,t, x, y, w, h, wx, wy, s
# Tracklets => 9 dim
# tracklets id, st_frame, ed_frame, st_vel(vec), ed_vel(vec), tracklet_len,


from model.seresnext import seresnext50_32x4d
# from torch.nn.utils.rnn import PackedSequence
class Cleaving(nn.Module):
    def __init__(self,wh=(128,256),track_len=120,id_len=1000,feat_size=2048, hidden_size=256, num_layers=4, dropout=0.2):
        super().__init__()
        # params
        self.track_len=track_len
        self.wh = wh

        # nn model
        self.feat_ex_module=seresnext50_32x4d(in_size=wh)

        self.siamModel = SiameseBiGRU(track_len,feat_size, hidden_size, num_layers, dropout).cuda()
        self.feat_fcn = nn.Linear(hidden_size, id_len)

        # self.srh_fcn = nn.Linear(track_len - 1, track_len - 1)
        self.srh_fcn = nn.Sequential(
            nn.Linear(track_len - 1, track_len - 1),
            nn.Sigmoid()
        )
        self.pur_fcn = nn.Sequential(
            nn.Linear(track_len - 1, 2),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_size,id_len)
        )
        self.width = wh[0]
        self.height = wh[1]

    def forward(self,x): # x.shape == batch,track, feat
        batch_in=x.shape[0]
        x=x.view(self.track_len,batch_in,-1,self.wh[0],self.wh[1])
        feat = []
        for j in range(batch_in):
            feat_=[]
            for i in range(self.track_len):
                feat_.append(self.feat_ex_module(x[i]))
            feat.append(torch.stack(feat_))
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

        phi_d = (phi_g_for[:,1:]-torch.flip(phi_g_back[:,1:],[0])).pow(2).sum(2).sqrt()

        srh_output = self.srh_fcn(phi_d)
        pur_output = self.pur_fcn(phi_d)
        return gru_output_f,gru_output_b,srh_output,pur_output,self.classifier(feat)

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
# class Reconnect(Base_Model):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#         print("init done")
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))
# class PPN(Base_Model):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)
#         print("init done")
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))

