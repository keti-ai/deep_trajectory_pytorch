import numpy as np
import matplotlib.pyplot as plt


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


P = getPositionEncoding(seq_len=4, d=4, n=4)
print(P)

# [[ 0.          1.          0.          1.        ]
#  [ 0.84147098  0.54030231  0.09983342  0.99500417]
#  [ 0.90929743 -0.41614684  0.19866933  0.98006658]
#  [ 0.14112001 -0.9899925   0.29552021  0.95533649]]

# P = getPositionEncoding(seq_len=1000, d=512, n=10000)
cax = plt.matshow(P)
import visdom
vis = visdom.Visdom()
vis.images(P)
plt.gcf().colorbar(cax)

'''

### Linde Plot ###
Y_data = torch.randn(5)
plt = vis.line (Y=Y_data)
# X값을 선언하지 않으면 0~1 사이의 값만 불러오게 됨

X_data = torch.Tensor([1,2,3,4,5]) # x축 명시 
plt = vis.line(Y=Y_data, X=X_data)

### Line Updtae ###
Y_append = torch.randn(1)
X_append = torch.Tensor([6])

vis.line(Y=Y_append, X=X_append, win=plt, update='append')

### multiple Line on single windows ###
num = torch.Tensor(list(range(0,10)))
num = num.view(-1,1)
num = torch.cat((num,num),dim=1)

plt = vis.line(Y=torch.randn(10,2), X = num)


### Line Info ###
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', showlegend=True))
plt = vis.line(Y=Y_data, X=X_data, opts = dict(title='Test', legend = ['1번'],showlegend=True))
plt = vis.line(Y=torch.randn(10,2), X = num, opts=dict(title='Test', legend=['1번','2번'],showlegend=True))

'''