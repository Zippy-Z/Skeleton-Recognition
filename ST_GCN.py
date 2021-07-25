import torch
import numpy as np
import glob
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import glob
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn as nn
import torch.nn.functional as F


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
        ndata = np.zeros([1, 3, seq_min, 17, 2])
        for j in range(seq_min):
            ndata[:, :, j, :, :] = temp_data[i][:, :, j, :, :]
        temp_data[i] = ndata
    # 数据增强(平移，添加90条第五类)
    old_len = len(temp_data)
    if is_train_data:
        for i in range(10):
            for j in range(9):
                shift = np.random.rand()
                d = np.zeros([1, 3, seq_min, 17, 2])
                if temp_data[400 + i][0, 0, 0, 0, 1] == 0:
                    d[:, :, :, :, :] = temp_data[400 + i]
                    d[:, 0, :, :, 0] = d[:, 0, :, :, 0] + shift
                    d[:, 2, :, :, 0] = d[:, 2, :, :, 0] + shift
                else:
                    d[:, 0, :, :, :] = temp_data[400 + i][:, 0, :, :, :] + shift
                    d[:, 2, :, :, :] = temp_data[400 + i][:, 2, :, :, :] + shift
                temp_data.append(d)
                temp_label.append(4)
                temp_seq_len.append(temp_seq_len[400 + i])
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
    nn_data = np.zeros([len(temp_data), 3, seq_min, 17, 1])
    for i in range(len(temp_data)):
        nn_data[i, :, :, :, 0] = temp_data[i][:, :, :, :, 0]
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


def cal_acc(out, label):
    out = out.detach().numpy()
    label = label.detach().numpy()
    if np.argmax(out) == label:
        return 1
    else:
        return 0


num_node = 17
self_link = [(i, i) for i in range(num_node)]
inward = [
    (10, 8), (8, 6), (9, 7), (7, 5),  # arms
    (15, 13), (13, 11), (16, 14), (14, 12),  # legs
    (11, 5), (12, 6), (11, 12), (5, 6),  # torso
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)  # nose, eyes and ears
]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import os
#
#     A = Graph('spatial').get_adjacency_matrix()
#     for i in A:
#         plt.imshow(i, cmap='gray')
#         plt.show()
#     print(A)


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = x.float()
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class,
                 edge_importance_weighting, **kwargs):  # graph_args,
        super().__init__()

        # load graph
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # self.sg = nn.Sequential(
        #     nn.Sigmoid()
        # )

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        # x = self.sg(x)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        # 输入(N,C,T,V)对应数据前四维
        res = self.residual(x)
        x, A = self.gcn(x, A)  # 中间变量x是(N,C,T,V)，K代表stgcn输入的ksize的第二维
        x = x.double()
        x = self.tcn(x) + res

        return self.relu(x), A


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 创建时序容器
            nn.Conv2d(256, 512, kernel_size=(3, 4)),
            nn.BatchNorm2d(512),  # 归一化
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # #
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
            nn.Linear(512*7*5, 5),
            # nn.Linear(256 * 12 * 17, 5),
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

def feature_resize(fm):
    n_fm = np.zeros([fm.shape[0], fm.shape[1],fm.shape[2],fm.shape[3]])
    n_fm = fm[:,:,:,:,0]
    return n_fm


path = 'D:\锴子\人工智能综合实验\期末大作业\人体姿态序列分类\人体姿态序列分类/data'
split = 'train'
data1, seq_len1, label1 = load_data(path, split)
split = 'test'
data2, seq_len2, label2 = load_data(path, split)
seq_min = min_seq_len(seq_len1, seq_len2)
# print(seq_min)
train_min_len, train_data, train_seq_len, train_label1 = data_packing(data1, seq_len1, label1, seq_min, True)
train_label = label_process(train_label1)
test_min_len, test_data, test_seq_len, test_label1 = data_packing(data2, seq_len2, label2, seq_min, False)
test_label = label_process(test_label1)
STGCN = Model(in_channels=3, num_class=5, edge_importance_weighting=False)
train_data = Variable(train_data)
_, train_fm = STGCN.extract_feature(train_data)
test_data = Variable(test_data)
_, test_fm = STGCN.extract_feature(test_data)  #   [size, 256,seq_len(12), 17]
train_nfm = feature_resize(train_fm)
test_nfm = feature_resize(test_fm)
# for i in range(test_nfm.shape[0]):
#     print(test_nfm[i,:,:,:])
# train_fm = STGCN(train_data)
# test_fm = STGCN(test_data)




# 定义一些超参数
batch_size = 32
learning_rate = 5e-4
num_epoches = 10
Train_data = TensorDataset(train_nfm, train_label)
Train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=False)
Test_data = TensorDataset(test_nfm, test_label)
Test_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=False)   #   [16, 3, seq_len(46), 17]

# 选择模型
cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

# 定义损失函数和优化器
Criterion = nn.CrossEntropyLoss()
Optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)

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
        out = cnn(d)
        loss = Criterion(out, label)
        print_loss = loss.data.item()
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

# 模型评估
Pred = np.zeros([117, 5])
count = 0
cnn.eval()
for data in Test_loader:
    d, label = data
    d = Variable(d)
    # print(d)
    if torch.cuda.is_available():
        d = d.cuda()
        label = label.cuda()
    label = label.long()
    out = cnn(d)
    print(out)
    for xy in range(list(out.size())[0]):
        for i in range(5):
            Pred[count, i] = out[xy, i]
        count += 1
    loss = Criterion(out, label)
Acc = Get_Acc(Pred, test_label)
# print(Pred)
print('测试集正确率: ' + str(100 * Acc) + '%')



