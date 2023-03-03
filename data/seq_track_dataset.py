import os

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


from PIL import Image


class Bbox():
    def __init__(self, url, id_, cam, proj, ext_, frame):
        self.id = int(id_)
        self.cam = int(cam[1:])
        self.proj = proj
        self.ext_ = ext_
        self.frame = int(frame)
        self.url = url

    # def __repr__(self):
    #     return '_'.join([str(self.id),str(self.cam),self.proj,self.ext_,str(self.frame)])
    def __repr__(self):
        return self.url

    def get_frame(self):
        return self.frame

    def get_id(self):
        return self.id

    def get_cam(self):
        return self.cam

    def get_proj(self):
        return self.proj

    def __lt__(self, other):
        if type(other) != Bbox:
            return self.get_frame() < other
        else:
            return self.get_frame() < other.get_frame()

    def __le__(self, other):
        return self.get_frame() <= other.get_frame()

    def __gt__(self, other):
        return self.get_frame() > other.get_frame()

    def __ge__(self, other):
        return self.get_frame() <= other.get_frame()

    def __eq__(self, other):
        return self.get_frame() == other.get_frame()


tracklet_id = 0


class Tracklet():
    def __init__(self, id_, cam, proj, init_box, duration, length):
        self.length = length
        # tracklet_id +=1
        self.tracklet_id = tracklet_id
        self.id = int(id_)
        self.cam = int(cam)
        self.proj = proj
        self.dur = duration
        # self.ext_=ext_
        # self.frame=int(frame)
        # self.url=url
        if type(init_box)==Bbox:
            self.bbox = [init_box]
        else:
            self.bbox = init_box

    def get_tr_id(self):
        return self.tracklet_id
    def __getitem__(self,idx):
        return self.bbox[idx]
    def __len__(self):
        return len(self.bbox)

    def add_bbox(self, bbox):
        if bbox.get_id() != self.id or bbox.get_cam() != self.cam or len(self) >= 120:
            return False
        else:
            if self.bbox[-1] < bbox and self.bbox[-1].get_frame() + self.dur > bbox:
                self.bbox.append(bbox)
            else:
                return False
        self.bbox.sort()
        return True
    def add_bboxes(self,bboxes):
        self.bbox=self.bbox+bboxes
    def get_id(self):
        return self.id

    def get_bboxes(self):
        return self.bbox




class TrackDataSet(Dataset):
    def __init__(self,root_dir,track_length):
        self.root_dir = root_dir
        self.track_len = track_length

        self.tracks=[]
        self.transforms=transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((224,224))])

        self.load_tracks()
        print("Load Track done")
        #
        # self.data = np.random.rand(10,120,256,128,3)
        # self.label = np.random.randint(0,1000,(10,120))

    def __len__(self):
        return len(self.tracks)
    def __getitem__(self, idx): # idx == iter
        '''
        idx = [ 4,6,2,1] // batch with indexes
        self.data = np.random.rand(10,120,256,128,3)
        self.label = np.random.randint(0,1000,(10,120))
        track == 120*bbox
        '''
        in_batch=idx.__len__()
        in_tracks=[]
        for i in range(len(idx)):
            in_tracks.append(self.tracks[idx[i]])

        data_=[]
        # labels = np.random.randint(0,23,(in_batch,120))

        # print("idx :",idx)
        labels = []
        imgs = []
        for i in range(len(idx)):
            label_ = []
            imgs_ = []
            for tr in in_tracks:
                label_=[]
                imgs_=[]
                for bb in tr:
                    label_.append(bb.get_id())

                    with Image.open(os.path.join(self.root_dir,str(bb))) as image:
                        if self.transforms:
                            image = self.transforms(image)
                        imgs_.append(image)

            labels.append(label_)
            imgs.append(torch.stack(imgs_))

        return torch.Tensor(torch.stack(imgs)).float().cuda(),\
            torch.Tensor(labels).cuda().type(torch.cuda.LongTensor)

    # ToDo tracklet 만들기
    # method load_tracks 에서 tracklet 을 정의해고 해당 되는  bbox 링크만 정의
    # image 로드는 __get_item__에서 실행
    # make Track List
    def load_tracks(self):
        track = []
        list_bbox = os.listdir(self.root_dir)
        list_bbox.sort()


        bbox_list = []
        for li in list_bbox:
            bbox_list.append(Bbox(li, *li.split(".")[0].split("_")))

        bbox_list.sort()
        tracklet_id = 0
        length = 120
        delay = 50  # duration == 프레임간 같은 트랙이라고 용인할 frame 차이
        tracklets = [
            Tracklet(bbox_list[0].get_id(), bbox_list[0].get_cam(), bbox_list[0].get_proj(), bbox_list[0], delay,
                     length)]
        print("Make Initial Tracklets ...")
        print("Tracklet Delay Frame : ", delay)
        for bb in bbox_list:
            is_add = False
            for tracklet in tracklets:
                if tracklet.add_bbox(bb):
                    is_add = True
                    break
            if is_add:
                continue
            tracklets.append(Tracklet(bb.get_id(), bb.get_cam(), bb.get_proj(), bb, delay, length))
            tracklet_id += 1
        print("Gen initial Tracklet : ", tracklet_id)

        track_120=[]
        for tr in tracklets:
            if len(tr) == 120:
                track_120.append(tr)
        indee=0
        pos_tracks = []
        negee_tracks = []
        neg_tracks= []
        for tr in track_120:
            if indee>=200:
                in_boxes=tr.get_bboxes()
                negee_track.append(Tracklet(in_boxes[0].get_id(), in_boxes[0].get_cam(), in_boxes[0].get_proj(), in_boxes[:60], delay, length))
                negee_track.append(
                    Tracklet(in_boxes[0].get_id(), in_boxes[0].get_cam(), in_boxes[0].get_proj(), in_boxes[60:], delay,
                             length))
            else:
                pos_tracks.append(tr)
        while len(negee_tracks)!=0:
            in_tr = negee_tracks.pop(0)
            pop_ind = 0
            get_=False
            for i, tr in enumerate(negee_tracks):
                if in_tr.get_id() != tr.get_id():
                    get_=True
                    in_tr.add_bboxes(tr)
                    pop_ind = i
                    negee_tracks.pop(pop_ind)
                    break
            if get_:
                neg_tracks.append(in_tr)



        self.tracks=pos_tracks+neg_tracks
        #
        # for filename in sorted(os.listdir(self.root_dir)):
        #     if filename.endswith('.jpg') or filename.endswith('.png'):
        #         gt_track,cam,proj,_,frame=filename.split(".")[0].split("_")
        #
        #         image_path = os.path.join(self.root_dir, filename)
        #
        #         #
        #
        #         track.append(image_path)
        #
        #         if len(track) == self.track_len:
        #             self.tracks.append(track)
        #             track = []
        #
        # if track:
        #     self.tracks.append(track)