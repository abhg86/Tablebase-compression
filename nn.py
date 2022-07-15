from ctypes import sizeof
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import f1_score
from torchmetrics import ConfusionMatrix
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy.random as rd
import numpy as np
import random
import math
# import main
import chess
import chess.syzygy
import csv


def create_board(pieces,places, color):
    board = chess.Board()
    board.clear()

    for i in range(len(pieces)):
        board.set_piece_at(places[i], chess.Piece.from_symbol(pieces[i]))

    if color == 'b':
        board.turn = chess.BLACK        # safe to do according to the doc
    
    return board

def random_board(pieces):
    places = random.sample(chess.SQUARES, len(pieces))
    color = random.choice(['w', 'b'])
    board = create_board(pieces, places, color)
    if not board.is_valid() : 
        return random_board(pieces)
    else:
        return board

def board_to_tensor(board):
    fen = board.fen()
    tensor = torch.zeros((13,8,8))
    blank = False
    for i, letter in enumerate(fen):
        a,b = divmod(i,8)
        if letter == ' ' and blank:
            break
        elif letter == ' ' :
            ()

        elif letter == 'w' and blank :
            tensor[12,:,:] = torch.zeros((8,8))
        elif letter == 'b' and blank :
            tensor[12,:,:] = torch.ones((8,8))

        elif letter == 'K':
            tensor[0,a,b] = 1
        elif letter == 'Q':
            tensor[1,a,b] = 1
        elif letter == 'R':
            tensor[2,a,b] = 1
        elif letter == 'N' :
            tensor[3,a,b] = 1
        elif letter == 'B':
            tensor[4,a,b] = 1
        elif letter == 'P':
            tensor[5,a,b] = 1
        
        elif letter == 'k':
            tensor[6,a,b] = 1
        elif letter == 'q':
            tensor[7,a,b] = 1
        elif letter == 'r':
            tensor[8,a,b] = 1
        elif letter == 'n':
            tensor[9,a,b] = 1
        elif letter == 'b':
            tensor[10,a,b] = 1
        elif letter == 'p':
            tensor[11,a,b] = 1
    return tensor

def nn_train_set(nb, nb_pieces, pieces=None):
    Xtr, Ytr = [], []
    with chess.syzygy.open_tablebases("wdl") as tablebase :
        for i in range(nb):
            if not pieces :
                pieces = rd.choice(['Q','q','R','R','r','r','B','B','b','b','N','N','n','n','P','p','P','p','P','p','P','p','P','p','P','p','P','p','P','p'], nb_pieces -2, replace=False)
                pieces = list(pieces)
                pieces += ['k','K']
            board = random_board(pieces)
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

def nn_train_setv2(nb, nb_pieces, pieces=None):
    Xtr, Ytr = [], []
    with chess.syzygy.open_tablebases("wdl") as tablebase :
        i=0
        m2 = 0
        p2 = 0
        zer = 0
        while i < nb:
            if not pieces :
                pieces = rd.choice(['Q','q','R','R','r','r','B','B','b','b','N','N','n','n','P','p','P','p','P','p','P','p','P','p','P','p','P','p','P','p'], nb_pieces -2, replace=False)
                pieces = list(pieces)
                pieces += ['k','K']
            board = random_board(pieces)
            result = tablebase.probe_wdl(board)
            if (result == -2 and m2 <= 333) or (result == 2 and p2 <= 333) or (result == 0 and zer <= 333):
                i+= 1

                Xtr += [board_to_tensor(board)]

                if result == -2:
                    m2 += 1
                    result = 0
                elif result == 0:
                    zer += 1
                    result = 1
                else :
                    p2 += 1
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
        self.conv1 = nn.Conv2d(13, 64, kernel_size = 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)

    def forward(self, s):
        s = s.view(-1, 13, 8, 8)  # batch_size x channels x board_x x board_y
        s = self.conv1(s)
        s = F.relu(self.bn1(s))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=64, planes=64, stride=1, downsample=None):
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

class MobileBlock(nn.Module):
    def __init__(self, inplanes=64, stride=1, downsample=None):
        super(MobileBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 6 *inplanes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(6*inplanes)
        self.conv2 = nn.Conv2d(6* inplanes, 6*inplanes, kernel_size=3, stride=stride, padding=1, bias=False, groups = 6*inplanes)
        self.bn2 = nn.BatchNorm2d(6*inplanes)
        self.conv3 = nn.Conv2d(6 * inplanes, inplanes, kernel_size = 1, stride = stride, padding = 0, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = F.relu(out)
        return out

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64*8*8, 3)
        self.losoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,s):
        # s = self.avg_pool(s)
        s = torch.flatten(s, 1)
        s = self.fc(s)
        s = self.losoftmax(s)            #return a logproba
        return s
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(20):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(20):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
        

def train(net, dataset, device, criterion, optimizer, scheduler, epoch_start=0, epoch_stop=1):
    net.train()
    
    train_set = myDataset(dataset)
    # train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    total_loss = 0.0
    for epoch in range(epoch_start, epoch_stop):
        boards = []
        results = []
        for data in train_set:
            board, result = data[0].to(device), data[1].to(device)          #j'ignore si le to(device) est nécessaire ici, si non on peut se débarasser de MyDataset
            boards += [board]
            results += [result]
        in_boards = torch.stack(boards, dim = 0)
        out_results = torch.stack(results, dim = 0)
        in_boards.to(device)
        out_results.to(device)

        optimizer.zero_grad()
        pred = net(in_boards)
        out_results = torch.flatten(out_results)
        loss = criterion(pred, out_results)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / (epoch_stop - epoch_start)

def partial_bounds(net, Xt, Yt, N, device):
    nb_right_ans = 0
    log_proba = []
    preds = torch.zeros(len(Yt), dtype=torch.int32)
    preds.to(device)
    targets = torch.zeros(len(Yt), dtype=torch.int32)
    targets.to(device)
    cross_ent = nn.CrossEntropyLoss()
    train_set = myDataset([Xt, Yt])
    net.eval()
    # x_test = torch.tensor([0])
    for i, (x, y)  in enumerate(train_set):
        y = y.item()
        board = x.to(device)
        x_values, x_labels = torch.topk(net(board),3)
        x_values.to(device)
        x_labels.to(device)
        # print(torch.equal(x_test, x_labels))
        # x_test = x_labels

        if y == x_labels[0,0].item():
            nb_right_ans += 1
        
        preds[i] = x_labels[0,0].item()
        targets[i] = y

        logproba = x_values[0,x_labels.cpu().numpy()[0].tolist().index(y)].item()
        log_proba += [-logproba]


    log_proba = np.array(log_proba)
    n = len(Xt)
    acc = nb_right_ans / n
    moyenne = log_proba.mean()
    f1score = f1_score(preds, targets, num_classes = 3).item()
    confmat = ConfusionMatrix(num_classes = 3)
    conf_mat = confmat(preds, targets,)
    return ((1-acc)* N * (math.log(N)+2), N * moyenne, moyenne, log_proba.std(), acc, f1score, conf_mat)


def main_loop():
    net = ChessNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10000)
    current_loss = 0
    loss = []
    N = 15249024
    with open('test.csv', 'a', newline='') as fichiercsv:
        w = csv.writer(fichiercsv)
        w.writerow(['New data'])
        w.writerow(['Step', 'loss'])
        X, Y = nn_train_setv2(1000, 4)
        data = [X,Y]
        for i in range(10000):                                    #arbitrary, can be changed 
            
            current_loss = train(net, data, device, criterion, optimizer, scheduler)
            if i%10 == 0 :
                # Xt, Yt = nn_train_set(1000, 4)          #actually a test set but no verification is needed so train_set is more appropriate
                borne1, borne2, moy, ec_type, acc, f1score, conf_mat = partial_bounds(net, X, Y, N, device)
                print(i)
                print(acc)
                print(conf_mat)
                w.writerows([[i, 'current loss', current_loss], [i, 'bound1', borne1], [i, 'bound2', borne2], [i, 'mean', moy], [i, 'standard deviation', ec_type], [i, 'accuracy', acc], [i, 'f1 score', f1score], [i, 'confusion matrix', conf_mat],])


main_loop()
