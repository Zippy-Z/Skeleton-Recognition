from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import glob
import torch
import random

# 定义一些超参数
batch_size = 16
learning_rate = 1e-3
num_epoches = 30


# 去除随机性
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministric = True
torch.backends.cudnn.benchamark = False
torch.set_default_tensor_type(torch.DoubleTensor)


def load_data(path, split):
    temp_data = []
    temp_seq_len = []
    temp_label = []
    for i in range(5):
        label = '00' + str(i)
        filename = glob.glob(path + '/' + split + '/' + label + '/*.npy')
        for ind, file in enumerate(filename):
            data = np.load(file)
            temp_data.append(data)
            temp_seq_len.append(data.shape[2])
            temp_label.append(i)
    return temp_data, temp_seq_len, temp_label


def min_seq_len(train_seq_len, test_seq_len):
    seq_min_1 = train_seq_len[np.argmin(train_seq_len)]
    seq_min_2 = test_seq_len[np.argmin(test_seq_len)]
    if seq_min_1 < seq_min_2:
        return seq_min_1
    else:
        return seq_min_2


def data_packing(temp_data, temp_seq_len, temp_label, seq_min, is_train_data):
    # 统一时间序列长度，取最短值
    for i in range(len(temp_seq_len)):
        ndata = np.zeros([1,3,seq_min,17,2])
        for j in range(seq_min):
            ndata[:,:,j,:,:] = temp_data[i][:,:,j,:,:]
        temp_data[i] = ndata
    # 数据增强(平移，添加90条第五类)
    old_len = len(temp_data)
    if is_train_data:
        for i in range(10):
            for j in range(9):
                shift = np.random.rand()
                d = np.zeros([1,3,seq_min,17,2])
                if temp_data[400+i][0,0,0,0,1] == 0:
                    d[:,:,:,:,:] = temp_data[400+i]
                    d[:,0,:,:,0] = d[:,0,:,:,0] + shift
                    d[:, 2, :, :, 0] = d[:, 2, :, :, 0] + shift
                else:
                    d[:, 0, :, :, :] = temp_data[400+i][:, 0, :, :, :] + shift
                    d[:, 2, :, :, :] = temp_data[400 + i][:, 2, :, :, :] + shift
                temp_data.append(d)
                temp_label.append(4)
                temp_seq_len.append(temp_seq_len[400+i])
    # # 排序
    # order_ind = np.argsort(temp_seq_len)
    # n_data = []
    # n_seq_len = []
    # n_label = []
    # for i in range(len(temp_seq_len)):
    #     n_data.append(temp_data[order_ind[i]])
    #     n_seq_len.append(temp_seq_len[order_ind[i]])
    #     n_label.append(temp_label[order_ind[i]])
    # n_data.reverse()
    # n_seq_len.reverse()
    # n_label.reverse()
    nn_data = np.zeros([len(temp_data),3,seq_min,17])
    for i in range(len(temp_data)):
        nn_data[i, :, :, :] = temp_data[i][:, :, :, :, 0]
    min_len = seq_min
    nnn_data = torch.from_numpy(nn_data)
    nnn_data = nnn_data.double()
    return min_len, nnn_data, temp_seq_len, temp_label


def label_process(label):
    n_label = np.ndarray([len(label), ])
    for i in range(len(label)):
        n_label[i] = label[i]
    nn_label = torch.from_numpy(n_label)
    nn_label = nn_label.long()
    return nn_label


# def cal_acc(out, label):
#     out = out.detach().numpy()
#     label = label.detach().numpy()
#     if np.argmax(out) == label:
#         return 1
#     else:
#         return 0


# 使用pytorch搭建卷积神经网络  [batchsize, 3, 46,17] x
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 创建时序容器
            nn.Conv2d(3, 6, kernel_size=(3, 4)),
            nn.BatchNorm2d(6),  # 归一化
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(4, 8, kernel_size=3),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.layer4 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        self.fc = nn.Sequential(
            nn.Linear(6 * 22 * 7, 5),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def Get_Acc(out, label):
    num_correct = 0
    for i in range(label.shape[0]):
        max_index1 = np.argmax(out[i, :])
        print("第" + str(i) + "个：")
        print(label[i])
        print(np.argmax(out[i, :]))
        if max_index1 == label[i]:
            num_correct += 1
    Acc = num_correct / out.shape[0]
    return Acc
path = 'data'
split = 'train_enhance'
data1, seq_len1, label1 = load_data(path, split)
split = 'test'
data2, seq_len2, label2 = load_data(path, split)
seq_min = min_seq_len(seq_len1, seq_len2)
train_min_len, train_data, train_seq_len, train_label1 = data_packing(data1, seq_len1, label1, seq_min, True)
train_label = label_process(train_label1)
test_min_len, test_data, test_seq_len, test_label1 = data_packing(data2, seq_len2, label2, seq_min, False)
test_label = label_process(test_label1)

#   数据集封装
Train_data = TensorDataset(train_data, train_label)
Train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=False)
Test_data = TensorDataset(test_data, test_label)
Test_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False)

# 选择模型
Model = CNN()
if torch.cuda.is_available():
    Model = Model.cuda()

# 定义损失函数和优化器
Criterion = nn.CrossEntropyLoss()
Optimizer = optim.SGD(Model.parameters(), lr=learning_rate)

# 训练模型
for count in range(num_epoches):
    print("回合：" + str(count))
    for data in Train_loader:
        d, label = data
        d = Variable(d)
        if torch.cuda.is_available():
            d = d.cuda()
            label = label.cuda()
        else:
            d = Variable(d)
            label = Variable(label)
        label = label.long()
        out = Model(d)
        loss = Criterion(out, label)
        print_loss = loss.data.item()
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

# 模型评估
Pred = np.zeros([117, 5])
count = 0
Model.eval()
for data in Test_loader:
    d, label = data
    d = Variable(d)
    # print(d)
    if torch.cuda.is_available():
        d = d.cuda()
        label = label.cuda()
    label = label.long()
    out = Model(d)
    print(out)
    for xy in range(list(out.size())[0]):
        for i in range(5):
            Pred[count, i] = out[xy, i]
        count += 1
    loss = Criterion(out, label)
Acc = Get_Acc(Pred, test_label)
# print(Pred)
print('测试集正确率: ' + str(100 * Acc) + '%')