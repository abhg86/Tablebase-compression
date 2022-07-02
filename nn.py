from ctypes import sizeof
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy.random as rd
import main
import chess
import csv


def board_to_tensor(board):
    fen = board.fen()
    tensor = torch.zeros((8,8))
    
    for i, letter in enumerate(fen):
        a,b = divmod(i,8)
        if letter == ' ':
            break

        elif letter == 'K':
            tensor[a,b] = 6
        elif letter == 'Q':
            tensor[a,b] = 5
        elif letter == 'R':
            tensor[a,b] = 4
        elif letter == 'N' :
            tensor[a,b] = 3
        elif letter == 'B':
            tensor[a,b] = 2
        elif letter == 'P':
            tensor[a,b] = 1
        
        elif letter == 'k':
            tensor[a,b] = -6
        elif letter == 'q':
            tensor[a,b] = -5
        elif letter == 'r':
            tensor[a,b] = -4
        elif letter == 'n':
            tensor[a,b] = -3
        elif letter == 'b':
            tensor[a,b] = -2
        elif letter == 'p':
            tensor[a,b] = -1
    return tensor

def nn_train_set(nb, nb_pieces, pieces=None):
    Xtr, Ytr = [], []
    with chess.syzygy.open_tablebase("data/syzygy/wdl") as tablebase :
        for i in range(nb):
            if not pieces :
                pieces = rd.choice(['Q','q','R','R','r','r','B','B','b','b','N','N','n','n','P','p','P','p','P','p','P','p','P','p','P','p','P','p','P','p'], nb_pieces -2, replace=False)
                pieces = list(pieces)
                pieces += ['k','K']
            board = main.random_board(pieces)
            result = tablebase.probe_wdl(board)
            Xtr += [board_to_tensor(board)]

            if result == -2:
                result = 0
            elif result == 0:
                result = 1
            else :
                result = 2
            Ytr += [torch.tensor([result])]
    return Xtr, Ytr

class myDataset(Dataset):
    def __init__(self, data):
        self.data = data                            #data = [boards, results]
    
    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        board = self.data[0][idx]
        result = self.data[1][idx]

        return board, result


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size = 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 1, 8, 8)  # batch_size x channels x board_x x board_y
        s = self.conv1(s)
        s = F.relu(self.bn1(s))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.fc = nn.Linear(256*8*8, 3)
        self.losoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,s):
        s = torch.flatten(s, 1)
        s = self.fc(s)
        s = self.losoftmax(s).exp()             #advised by the doc
        return s
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
        

def train(net, dataset, device, epoch_start=0, epoch_stop=1):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)
    
    train_set = myDataset(dataset)
    # train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    total_loss = 0.0
    for epoch in range(epoch_start, epoch_stop):
        boards = []
        results = []
        for data in train_set:
            board, result = data[0].to(device), data[1].to(device)          #j'ignore si le to(device) est nécessaire ici, si non on peut se débarasser de MyDataset
            boards += board
            results += result
        in_boards = torch.stack(boards, dim = 0)
        out_results = torch.stack(results, dim = 0)
        in_boards.to(device)
        out_results.to(device)
        optimizer.zero_grad()

        pred = net(in_boards) 
        loss = criterion(pred, out_results)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / (epoch_stop - epoch_start)


def main_loop():
    net = ChessNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)
    current_loss = 0
    loss = []
    with open('data_nn.csv', 'a', newline='') as fichiercsv:
        w = csv.writer(fichiercsv)
        w.writerow(['New data'])
        w.writerow(['Step', 'loss'])
        for i in range(1000):                                    #arbitrary, can be changed 
            X, Y = nn_train_set(1000, 4)
            data = [X,Y]
            current_loss = train(net, data, device)
            if i%10 == 0 and i>20 :
                w.writerow([i, current_loss])
                loss += [current_loss]
                current_loss = 0
    return loss

loss = main_loop()
plt.plot(loss)
plt.show()