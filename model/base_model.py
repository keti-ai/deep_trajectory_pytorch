import torch.nn as nn
import torch.nn.functional as F

class Base_Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class Model_example(Base_Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        print("init done")
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

