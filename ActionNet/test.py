import torch
from modelss.action import Action
import pickle
import pandas as pd

def test():
    torch.randint()
    input = torch.randn(8 * 8, 64, 10, 10)  # (N*T,C_in,H_in,W_in)
    print(input.size())

    net = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    output1 = net(input)
    print(output1.size())

    net = Action(net, n_segment=8, shift_div=8)
    output2 = net(input)
    print(net)
    print(output2.size())


def pkl():
    path = '../data/sthv2_annotation/train.pkl'
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth', 1000000)
    print(type(data))
    print('---')
    print(data[0:3])
    print('---')


if __name__ == '__main__':
    pkl()
