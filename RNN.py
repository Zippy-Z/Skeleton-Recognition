# -*- coding: utf-8 -*-

import numpy as np
from visualise import visualise
from graph import Graph
import glob
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import random



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


def max_seq_len(train_seq_len, test_seq_len):
    seq_max_1 = train_seq_len[np.argmax(train_seq_len)]
    seq_max_2 = test_seq_len[np.argmax(test_seq_len)]
    if seq_max_1 > seq_max_2:
        return seq_max_1
    else:
        return seq_max_2



def data_packing(temp_data, temp_seq_len, temp_label, seq_max, is_train_data):
    # 补零
    for i in range(len(temp_seq_len)):
        ndata = np.zeros([1,3,seq_max,17,2])
        for j in range(temp_seq_len[i]):
            ndata[:,:,j,:,:] = temp_data[i][:,:,j,:,:]
        temp_data[i] = ndata
    # 数据增强(平移，添加90条第五类)
    old_len = len(temp_data)
    if is_train_data:
        for i in range(10):
            for j in range(9):
                shift = np.random.rand()
                d = np.zeros([1,3,seq_max,17,2])
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
    # 排序
    order_ind = np.argsort(temp_seq_len)
    n_data = []
    n_seq_len = []
    n_label = []
    for i in range(len(temp_seq_len)):
        n_data.append(temp_data[order_ind[i]])
        n_seq_len.append(temp_seq_len[order_ind[i]])
        n_label.append(temp_label[order_ind[i]])
    n_data.reverse()
    n_seq_len.reverse()
    n_label.reverse()
    nn_data = np.zeros([len(n_data), seq_max, 3*17*2])
    # 拉伸，第一个人的17x17y17z到第二个人的17x17y17z
    for i in range(len(n_data)):
        t_data = np.zeros([1,seq_max,3*17*2])
        count = 0
        for j in range(2):
            for k in range(3):
                for n in range(17):
                    t_data[0,:,count] = n_data[i][0,k,:,n,j]
                    count += 1
        nn_data[i,:,:] = t_data

    max_len = seq_max
    nnn_data = torch.from_numpy(nn_data)
    nnn_data = nnn_data.float()
    return max_len, nnn_data, n_seq_len, n_label


def label_process(label):
    # 转成one-hot
    # n_label = np.zeros([len(label), 5])
    # for i in range(len(label)):
    #     n_label[i][label[i]] = 1

    n_label = np.ndarray([len(label), 1])
    for i in range(len(label)):
        n_label[i][0] = label[i]
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


import torch


class RNN_V1(torch.nn.Module):
    def __init__(self, hidden_size, out_size, n_layers=1, batch_size=1):
        super(RNN_V1, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了 BATCH FIRST
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

        # 加了一个线性层，全连接
        # self.out = torch.nn.Linear(hidden_size, out_size)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, out_size),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, word_inputs, hidden):
        # -1 是在其他确定的情况下，PyTorch 能够自动推断出来，view 函数就是在数据不变的情况下重新整理数据维度
        # batch, time_seq, input

        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)
        # hidden 就是上下文输出，output 就是 RNN 输出
        output, hidden = self.gru(inputs, hidden)

        output = self.out(output)

        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:, -1, :]

        return output, hidden

    def init_hidden(self):
        hidden = torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden


def _test_rnn_rand_vec(seq_max):
    encoder_test = RNN_V1(seq_max, 5, batch_size=1)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(encoder_test.parameters(), lr=1e-2, momentum=0.9)
    epoches = 10
    accs = np.zeros([10,])

    for j in range(epoches):
        print("回合:" + str(j))
        for i in range(train_data.shape[0]):
            # if j == 0:
            rnn_v1 = encoder_test.init_hidden()
        # input_data = torch.autograd.Variable(_xs[i])
        # output_labels = torch.autograd.Variable(torch.LongTensor([_ys[i]]))
            input_data = train_data[i]
            output_labels = train_label[i]
            input_lengths = train_seq_len[i]
        # print(output_labels)
            encoder_outputs, rnn_v1 = encoder_test(input_data, rnn_v1)

            loss = criterion(encoder_outputs, output_labels)
            accs[j] += cal_acc(encoder_outputs, output_labels)
            print("loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accs[j] /= train_data.shape[0]
    for i in range(10):
        print("第" + str(i) + "回合正确率:" + str(accs[i]))
    encoder_test.eval()
    accs = 0
    for i in range(test_data.shape[0]):
        rnn_v1 = encoder_test.init_hidden()
        input_data = test_data[i]
        output_labels = test_label[i]
        pred, rnn_v1 = encoder_test(input_data, rnn_v1)
        accs += cal_acc(pred, output_labels)
    accs /= test_data.shape[0]
    print("测试集正确率:" + str(accs))
    return encoder_test


# 去除随机性
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministric = True
torch.backends.cudnn.benchamark = False
# torch.set_default_tensor_type(torch.DoubleTensor)

path = 'D:\锴子\人工智能综合实验\期末大作业\人体姿态序列分类\人体姿态序列分类\data'
split = 'train'
data1, seq_len1, label1 = load_data(path, split)
split = 'test'
data2, seq_len2, label2 = load_data(path, split)
seq_max = max_seq_len(seq_len1, seq_len2)
train_max_len, train_data, train_seq_len, train_label1 = data_packing(data1, seq_len1, label1, seq_max, True)
train_label = label_process(train_label1)
test_max_len, test_data, test_seq_len, test_label1 = data_packing(data2, seq_len2, label2, seq_max, False)
test_label = label_process(test_label1)




trained_rnn = _test_rnn_rand_vec(seq_max)

