
import torch
from model.post_net import SiameseBiGRU, Cleaving

# Logging
import visdom
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

            val = [li if type(li)==int or type(li)==float else li.item() for li in val]
            self.vis.line(Y=[val],X=[x_],win=win_name, update="append" ,opts=dict(title=title, legend=self.legends,showlegend=True))
        else:
            self.vis.line(Y=[val if type(val)==int or type(val)==float else val.item()], X=[x_], win=win_name, update="append",opts=dict(title=title))

        self.wins.append(win_name)
        self.wins=list(set(self.wins))
        self._x=x_
    def vis_img(self,win_name,img):
        if len(img.shape)>3:
            self.vis.images(img,win=win_name)
        else:
            self.vis.image(img,win=win_name)

# Define the hyperparameters


from tqdm import tqdm
from data.seq_track_dataset import TrackDataSet
from data.dt_data_loader import base_collate_fn, BaseDataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler,SubsetRandomSampler,RandomSampler
import torch.optim as optim
import time
from model.deep_track_loss import FocalLoss



# Train the model

import os
if __name__ == "__main__":
    # vis_log=VisLogger(title="Test_loss",env_="main6",legends=["feat_loss","srh_loss","pur_loss","cls_loss","loss"])

    EXP_NAME="230313_1_srh_weight_cls_resume_141"
    args_resume = True

    ENV_NAME="main_"+EXP_NAME
    vis_log=VisLogger(title="Test_loss",env_=ENV_NAME,legends=["feat_loss","srh_loss","pur_loss","loss"])
    epochs = 1000

    feat_size = 2048  # size of input features
    hidden_size = 128
    num_layers = 4
    dropout = 0.2
    learning_rate = 0.001

    # Tracklet info
    track_len_in = 120
    # Create the model
    if args_resume:
        model = torch.load("/media/syh/hdd/checkpoints/deep_trajectory_230308_1_with_search_cls_w_loss/141_model.pth")
        # model = Cleaving(track_len=track_len_in, id_len=22, feat_size=feat_size, hidden_size=hidden_size,
        #                  num_layers=num_layers, dropout=dropout).cuda()
        # model.load_state_dict(model_data['model_state_dict'])
        # learning_rate = learning_rate**(141)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer.load_state_dict(model_data['optimizer_state_dict'])
        learning_rate =
        start_epoch = 141 + 1
        model.train()
    else:
        model = Cleaving(track_len=track_len_in, id_len=22, feat_size=feat_size, hidden_size=hidden_size,
                         num_layers=num_layers, dropout=dropout).cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start_epoch = 1

    # Define the loss function and optimizer
    lam_feat = 0.33
    lam_search = 0.33
    lam_pur = 0.33
    #lam_cls = 0.2

    # search weight
    # srh_weight_cls = [0.016, 1.984]
    srh_weight_cls = [1, 1]

    #criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_gru = torch.nn.CrossEntropyLoss()
    # criterion_srh = torch.nn.L1Loss()
    criterion_srh = FocalLoss(weight=torch.Tensor(srh_weight_cls).cuda())

    criterion_pur = torch.nn.BCELoss()

    # cleav_loss = lam_feat*criterion_gru + lam_search*criterion_srh + lam_pur * criterion_pur

    # basedataset = BaseDataSet()

    trackdataset = TrackDataSet(root_dir="/media/syh/ssd2/data/ReID/bounding_box_train",track_length=120)
    batch_size_ = 1

    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=True)

    # dataloader = BaseDataLoader(
    #     shuffle=True,
    #     dataset=trackdataset,
    #     collate_fn=base_collate_fn,
    #     batch_size=batch_size_,
    #     sampler=BatchSampler(
    #         SequentialSampler(trackdataset),
    #         batch_size=batch_size_,
    #         drop_last=False
    #     )
    # )
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
    iter = 0
    acc = 0
    best_acc = 0
    for epoch in range(start_epoch,epochs):
        running_loss = 0.0

        iter_epoch=0

        acc_srh = 0
        acc_gru = 0
        acc_gru_f = 0
        acc_gru_b = 0
        acc_pur_ = 0
        acc_cls = 0
        #acc_cls_ = 0

        acc_srh = 0
        acc_srh_ = 0

        # srh_counter = 0
        loss_sum = 0
        loss_gru_ = 0
        loss_srh_ = 0
        loss_pur_ = 0

        for i, data in enumerate(dataloader):
            iter+=1
            iter_epoch+=1
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
            # gru_out_f, gru_out_b, srh_out, pur_out,cls_out = model(torch.stack(tracklets))
            gru_out_f, gru_out_b, srh_out, pur_out = model(torch.stack(tracklets))

            label = torch.stack(labels)

            # loss = criterion(output1 - output2, label.float())
            loss_ = 0
            loss_gru = 0
            loss_srh = 0
            loss_pur = 0
            #loss_cls = 0


            for j in range(batch_size):
                # ToDo search network target 설정
                loss_gru += criterion_gru(gru_out_f[j], label[j]) + \
                            criterion_gru(gru_out_b[j], label[j])
                loss_pur += criterion_pur(pur_out[j], torch.Tensor([int(len(torch.unique(label))==1),int(len(torch.unique(label))!=1)]).cuda())
                #if int(len(torch.unique(label))!=1):
                srh_target=torch.Tensor([label[j][i] != label[j][i + 1] for i in range(len(label[j]) - 1)]).cuda()
                loss_srh += criterion_srh(srh_out[j].view(track_len_in-1,-1), srh_target)
                #loss_cls += criterion_cls(gru_out_f[j], label[j])
            loss_ = lam_feat * loss_gru + lam_search * loss_srh + lam_pur * loss_pur# + lam_cls * loss_cls
            loss_gru_ += loss_gru
            loss_srh_ += loss_srh
            loss_pur_ += loss_pur
            # vis_log.vis_log(win_name="main/loss",x_=iter,val=[loss_gru,loss_srh,loss_pur,loss_])
            loss_.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Print statistics
            running_loss += loss_.item()
            loss_sum += loss_.item()
            # print('[Epoch %d, Iteration %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            if i % 30 == 29:
                #vis_log.vis_log(win_name="main/loss_cls", x_=iter, val=loss_cls, title="loss_cls")
                print('[Epoch %d, Iteration %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 30))
                running_loss = 0.0
            # acc_cls
            #_,pred_c=cls_out.data.cpu().topk(1)
            #acc_cls_ +=torch.sum(pred_c[0].t() == label.data.cpu())
            # acc_gru
            _,pred=gru_out_f.data.cpu().topk(1)
            _,pred_ = gru_out_b.data.cpu().topk(1)

            acc_gru_f+=torch.sum(pred[0].t() == label.data.cpu())
            acc_gru_b+=torch.sum(pred_[0].t() == label.data.cpu())

            # acc_pur
            _,pred_pur=pur_out.data.cpu().topk(1)
            acc_pur_+=int(pred_pur==int(len(torch.unique(label))!=1))
            # acc srh
            # acc_srh_ += torch.sum(srh_out[0].cpu().view(track_len_in-1,2).topk(1)[1].squeeze() == torch.Tensor([label[j][i] != label[j][i + 1] for i in range(len(label[j]) - 1)]))
            # 230308 대부분이 0이라 acc 값 자체가 높게 측정됨 1을 잘 찾는지로 수정
            #
            srh_pred = (srh_out[0].view(track_len_in - 1, 2).topk(1)[1] == 1)
            if int(len(torch.unique(label))!=1): # class 두개 이상
                if len(srh_pred.nonzero(as_tuple=True)[0]) == 0: # 발견못했을 때
                    acc_srh += 0
                else: # 발견했을 때
                    acc_srh_ += torch.sum(srh_pred.nonzero(as_tuple=True)[0] == (srh_target == 1).nonzero(as_tuple=True)[0])
            else : # 한개의 클래스
                if len(srh_pred.nonzero(as_tuple=True)[0]) == 0:
                    acc_srh+=1

        #acc_cls = acc_cls_/(iter_epoch*track_len_in)
        acc_gru =(acc_gru_f+acc_gru_b)/(2*iter_epoch*track_len_in)
        acc_pur = acc_pur_/iter_epoch
        acc_srh = acc_srh_/iter_epoch
        loss_gru_/=iter_epoch
        loss_srh_/=iter_epoch
        loss_pur_/=iter_epoch
        loss_sum/=iter_epoch
        # vis_log.vis_img(win_name="main/input", img=data[0][0])
        # vis_log.vis_log(win_name="main/loss", x_=iter, val=[loss_gru, loss_srh, loss_pur, loss_cls, loss_])
        vis_log.vis_log(win_name=ENV_NAME+"/loss", x_=epoch, val=[loss_gru_, loss_srh_, loss_pur_, loss_sum])
        vis_log.vis_log(win_name=ENV_NAME+"/loss_Gru", x_=epoch, val=loss_gru_, title="loss_Gru")
        vis_log.vis_log(win_name=ENV_NAME+"/loss_srh", x_=epoch, val=loss_srh_, title="loss_srh")
        vis_log.vis_log(win_name=ENV_NAME+"/loss_pur", x_=epoch, val=loss_pur_, title="loss_pur")

        vis_log.vis_log(win_name=ENV_NAME+"/acc_Gru", x_=epoch, val=acc_gru,title="acc_Gru")
        vis_log.vis_log(win_name=ENV_NAME+"/acc_srh", x_=epoch, val=acc_srh,title="acc_srh")
        vis_log.vis_log(win_name=ENV_NAME+"/acc_pur", x_=epoch, val=acc_pur,title="acc_pur")
        #vis_log.vis_log(win_name="main/acc_cls", x_=epoch, val=acc_cls,title="acc_cls")
            # acc_srh
        FILENAME_MODEL = "%d_model.pth" % (epoch)
        FILENAME_DICT = "%d_model_dict.pt" % (epoch)
        FILENAME_ALL = "%d_all.tar" % (epoch)

        PATH_DIR = "/media/syh/hdd/checkpoints/deep_trajectory_"+EXP_NAME
        if not os.path.exists(PATH_DIR):
            os.makedirs(PATH_DIR)

        # PATH_MODEL = os.path.join(PATH_DIR, FILENAME_MODEL)
        # PATH_DICT = os.path.join(PATH_DIR, FILENAME_DICT)
        PATH_ALL = os.path.join(PATH_DIR, FILENAME_ALL)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH_ALL)

        # torch.save(model, PATH)

        scheduler.step()