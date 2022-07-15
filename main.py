import chess
import chess.syzygy
import random as rd
import math
import numpy as np 
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import sklearn.metrics as metrics
# from itertools import permutations

def create_board(pieces,places, color):
    board = chess.Board()
    board.clear()

    for i in range(len(pieces)):
        board.set_piece_at(places[i], chess.Piece.from_symbol(pieces[i]))

    if color == 'b':
        board.turn = chess.BLACK        # safe to do according to the doc
    
    return board

def random_board(pieces):
    places = rd.sample(chess.SQUARES, len(pieces))
    color = rd.choice(['w', 'b'])
    board = create_board(pieces, places, color)
    if not board.is_valid() : 
        return random_board(pieces)
    else:
        return board

def pieces_score(board):
    description = board.fen()
    score = [0,0,0,0,0,0,0,0,0,0]
    for i in description:
        if i == ' ':
            break

        elif i == 'Q':
            score[0] += 1
        elif i == 'R':
            score[1] += 1
        elif i == 'N' :
            score[2] += 1
        elif i == 'B':
            score[3] += 1
        elif i == 'P':
            score[4] += 1
        
        elif i == 'q':
            score[5] += 1
        elif i == 'r':
            score[6] += 1
        elif i == 'n':
            score[7] += 1
        elif i == 'b':
            score[8] += 1
        elif i == 'p':
            score[9] += 1
    
    return score

def nb_wmoves(board):
    color = board.turn
    board.turn = chess.WHITE
    score = board.legal_moves.count()
    board.turn = color
    return score

def nb_bmoves(board):
    color = board.turn
    board.turn = chess.BLACK
    score = board.legal_moves.count()
    board.turn = color
    return score

def all_pieces(board, color):
    '''return a set of squares occupied by the pieces of the given color'''
    return board.pieces(chess.PAWN, color).union(board.pieces(chess.KING, color).union(board.pieces(chess.QUEEN, color).union(board.pieces(chess.ROOK, color).union(board.pieces(chess.BISHOP, color).union(board.pieces(chess.KNIGHT, color))))))


def nb_wattack(board):
    count = 0
    for square in all_pieces(board, chess.BLACK):
        count += len(board.attackers(chess.WHITE, square))
    return count

def nb_battack(board):
    count = 0
    for square in all_pieces(board, chess.WHITE):
        count += len(board.attackers(chess.BLACK, square))
    return count

def nb_wpin(board):
    count = 0
    for square in all_pieces(board, chess.BLACK):
        count += int(board.is_pinned(chess.BLACK, square))
    return count

def nb_bpin(board):
    count = 0
    for square in all_pieces(board, chess.WHITE):
        count += int(board.is_pinned(chess.WHITE, square))
    return count



def features(board):
    '''features are independant of the turn to play because the result to predict is'''
    pieces = pieces_score(board)
    return [board.turn, pieces[0], pieces[1], pieces[2], pieces[3], pieces[4], pieces[5], pieces[6], pieces[7], pieces[8], pieces[9], int(board.is_check()), nb_wmoves(board), nb_bmoves(board), nb_wattack(board), nb_battack(board), nb_wpin(board), nb_bpin(board)]

def train_set(nb, nb_pieces, pieces=None):
    Xtr, Ytr = [], []
    boards = []
    with chess.syzygy.open_tablebase("wdl") as tablebase :
        for i in range(nb):
            if not pieces :
                pieces = np.random.choice(['Q','q','R','R','r','r','B','B','b','b','N','N','n','n','P','p','P','p','P','p','P','p','P','p','P','p','P','p','P','p'], nb_pieces -2, replace=False)
                pieces = list(pieces)
                pieces += ['k','K']
            board = random_board(pieces)
            boards += [board]

            feature = features(board)
            result = tablebase.probe_wdl(board)
            Xtr += [feature]
            Ytr += [result]
    return Xtr, Ytr, boards

def partial_bounds(clf, Xt, Yt, N):
    nb_right_ans = 0
    log_proba = []
    for x,y in zip(clf.predict_proba(Xt),Yt):
        if y == 0:                            #because probe_wdl return -2, 0 or 2
            pos = 1
        elif y == -2:
            pos = 0
        else :
            pos = 2

        if pos == np.argmax(x):
            nb_right_ans += 1
         
        log_proba += [-math.log2(x[pos])]

    log_proba = np.array(log_proba)
    n = len(Xt)
    acc = nb_right_ans / n
    moyenne = log_proba.mean()
    return ((1-acc)* N * (math.log(N)+2), N * moyenne, moyenne, log_proba.std())

def main():
    N = 1677216                   # number of boards possible with 4 pieces
    clf = SGDClassifier(loss = 'log')
    with open('data3.csv', 'a', newline='') as fichiercsv:
        w = csv.writer(fichiercsv)
        w.writerow(['New data'])
        w.writerow(['Step', 'metric', 'value'])
        for i in range(10000):                                    #arbitrary, can be changed 
            X, Y, _ = train_set(1000, 4)
            clf.partial_fit(X, Y, classes = [-2,0,2])
            if i%10 == 0 and i>20 :                                        #i > 20 in order to avoid the log error    
                Xt, Yt, _ = train_set(1000, 4)          #actually a test set but no verification is needed so train_set is more appropriate
                borne1, borne2, moy, ec_type = partial_bounds(clf, Xt, Yt, N)

                Ypred = clf.predict(Xt)                                     # faster than doing list(map(np.argmax, y)) with y being clf.predict_proba(Xt) precalculted for partial bounds
                conf_mat = metrics.confusion_matrix(Yt, Ypred)

                accuracy = clf.score(Xt,Yt)

                w.writerows([[i, 'bound1', borne1], [i, 'bound2', borne2], [i, 'mean', moy], [i, 'standard deviation', ec_type], [i, 'accuracy', accuracy], [i, 'confusion matrix', conf_mat],])

        Xt, Yt, _ = train_set(10000, 4)         #actually a test set but no verification is needed so train_set is more appropriate
    
        borne1, borne2, moy, ec_type = partial_bounds(clf, Xt, Yt, N)

        Ypred = clf.predict(Xt)                                     # faster than doing list(map(np.argmax, y)) with y being clf.predict_proba(Xt) precalculted for partial bounds
        conf_mat = metrics.confusion_matrix(Yt, Ypred)

        accuracy = clf.score(Xt,Yt)
    
        w.writerows([['final', 'bound1', borne1], ['final', 'bound2', borne2], ['final', 'mean', moy], ['final', 'standard deviation', ec_type], ['final', 'accuracy', accuracy], ['final', 'confusion matrix', conf_mat],])


main()
