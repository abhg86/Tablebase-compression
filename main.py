import chess
import chess.syzygy
import chess.gaviota
import sklearn 
import random as rd

from sklearn.linear_model import LogisticRegression

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

X, Y, boards = train_set(1000, ['K', 'k', 'Q', 'n'])
Xt, Yt = test_set(200, ['K', 'k', 'Q', 'n'], boards)
print("x : ", Xt)
print("y : ", Yt)

clf = LogisticRegression(random_state=0).fit(X, Y)
print(clf.score(Xt, Yt))
