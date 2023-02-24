import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import Base_Model

# bbox => 10 dim
# cid, id ,t, x, y, w, h, wx, wy, s
# Tracklets => 9 dim
# tracklets id, st_frame, ed_frame, st_vel(vec), ed_vel(vec), tracklet_len,


class SiameseBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(SiameseBiGRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward_once(self, x):
        output, _ = self.gru(x)
        out = torch.mean(output, dim=0)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    def forward(self, input1, input2):
        out1 = self.forward_once(input1)
        out2 = self.forward_once(input2)
        return out1, out2

class Cleaving(Base_Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        print("init done")

    def forward(self, x): #
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
class Reconnect(Base_Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        print("init done")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
class PPN(Base_Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        print("init done")
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

