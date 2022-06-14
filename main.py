import chess
import chess.syzygy
import chess.gaviota
import sklearn 
import random as rd
import math
import time 

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from itertools import permutations

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

def generate_boards(pieces, n):
    for i in range(n):
        yield random_board(pieces)

def generate_all(pieces):
    for positions in permutations(chess.SQUARES, len(pieces)):
        board1 = create_board(pieces, positions, 'w')
        board2 = create_board(pieces, positions, 'w')
        if board1.is_valid():
            yield board1
        if board2.is_valid():
            yield board2 

def generate_all_solutions(pieces):
    with chess.syzygy.open_tablebase("data/syzygy/wdl") as tablebase :
        for board in generate_all(pieces):
            yield tablebase.probe_wdl(board)



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

def train_set(nb, pieces):
    Xtr, Ytr = [], []
    boards = []
    with chess.syzygy.open_tablebase("data/syzygy/wdl") as tablebase :
        for i in range(nb):
            board = random_board(pieces)
            boards += [board]

            feature = features(board)
            result = tablebase.probe_wdl(board)
            Xtr += [feature]
            Ytr += [result]
    return Xtr, Ytr, boards

def test_set(nb, pieces, trainset):
    Xte, Yte = [], []
    with chess.syzygy.open_tablebase("data/syzygy/wdl") as tablebase :
        i = 0
        while i < nb:
            board = random_board(pieces)
            if not board in trainset:
                i += 1
                feature = features(board)
                result = tablebase.probe_wdl(board)
                Xte += [feature]
                Yte += [result]
        tablebase.close()
    return Xte, Yte

# X, Y, boards = train_set(1000, ['K', 'k', 'Q', 'r'])
# Xt, Yt = test_set(200, ['K', 'k', 'Q', 'r'], boards)

# clf = LogisticRegression(random_state=0).fit(X, Y)
# board, result = Xt[0], Yt[0]
# print(clf.score(Xt,Yt))
# print(result)
# print(clf.predict_proba([board]))


def bounds(clf, pieces):
    nb_right_ans = 0
    n = 0

    somme = 0
    with chess.syzygy.open_tablebase("data/syzygy/wdl") as tablebase :
        for board in generate_all(pieces):
            n+=1
            pos = tablebase.probe_wdl(board)
            if pos == clf.predict([features(board)])[0]:
                nb_right_ans += 1

            if pos == 0:                            #because probe_wdl return -2, 0 or 2
                pos = 1
            elif pos == -2:
                pos = 0
            somme -= clf.predict_log_proba([features(board)])[0][pos]

    acc = nb_right_ans / n
    return ((1-acc)* n * (math.log(n)+2), somme) 

def n_naif(k):
    n=64
    for i in range(64-k+1, 64):
        n *= i
    return n


def partial_bounds(clf, Xt, Yt, N):
    nb_right_ans = 0
    somme = 0
    for x,y in zip(Xt,Yt):
        if y == clf.predict([x])[0]:
            nb_right_ans += 1
         
        if y == 0:                            #because probe_wdl return -2, 0 or 2
            pos = 1
        elif y == -2:
            pos = 0
        else :
            pos = 2
        somme -= clf.predict_log_proba([x])[0][pos]

    n = len(Xt)
    acc = nb_right_ans / n
    return ((1-acc)* N * (math.log(N)+2), N * 1/n * somme)

N = n_naif(4)
board_tot = []
clf = SGDClassifier(loss = 'log')
for i in range(100):                                    #arbitrary, can be changed 
    X, Y, _ = train_set(1000, ['K', 'k', 'Q', 'r'])
    clf.partial_fit(X, Y, classes = [-2,0,2])
    if i%10 == 0 and i>20 :                                        #i > 20 in order to avoid the log error    
        Xt, Yt, _ = train_set(1000, ['K', 'k', 'Q', 'r'])          #actually a test set but no verification is needed so train_set is more appropriate
        borne1, borne2 = partial_bounds(clf, Xt, Yt, N)
        print(i, " :")
        print("borne 1 : ", borne1)
        print("borne 2 : ", borne2)

Xt, Yt, _ = train_set(10000, ['K', 'k', 'Q', 'r'])         #actually a test set but no verification is needed so train_set is more appropriate

borne1, borne2 = partial_bounds(clf, Xt, Yt, N)
print()
print("final")
print("borne 1 : ", borne1)
print("borne 2 : ", borne2)
