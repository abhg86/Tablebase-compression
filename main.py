import chess
import sklearn 
import random as rd

###
### Implémentation à la main avant de se rendre compte que c'était faisable par des commandes
###

# def add_to_line(p, i, line):
#     first = line[0]

#     if len(line) == 1:
#         n = int(first)
#         pre = i -1
#         post = n - i

#         if pre != 0 : strpre = str(pre)
#         else : strpre = ''
#         if post != 0 : strpost = str(post)
#         else : strpost = ''

#         return strpre+p+strpost

#     if first in ['k', 'q', 'r', 'b', 'n', 'p', 'K', 'Q', 'R', 'B', 'N', 'P']:
#         return first + add_to_line(p, i-1, line[1:])
#     else :
#         n = int(first)
#         if i<=n :
#             return add_to_line(p, i, [first]) + line[1:]
#         else :
#             return first + add_to_line(p, i -n, line[1:])


# def create_board(pieces, places, color):
#     board_lines = ["8","8","8","8","8","8","8","8"]
#     for i in range(len(pieces)):
#         p = pieces[i]
#         x,y = places[i]%8, places[i]//8
#         board_lines[y] = add_to_line(p, x+1, board_lines[y])
#     board = board_lines[0]+'/'+board_lines[1]+'/'+board_lines[2]+'/'+board_lines[3]+'/'+board_lines[4]+'/'+board_lines[5]+'/'+board_lines[6]+'/'+board_lines[7]+' ' + color + ' - - 0 1'
#     return chess.Board(board)

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
        random_board(pieces)
    return board



def pieces_score(board):
    description = board.fen()
    score = 0
    for i in description:
        if i == ' ':
            break

        elif i == 'Q':
            score += 9
        elif i == 'R':
            score += 5
        elif i == 'N' or i == 'B':
            score += 3
        elif i == 'P':
            score += 1
        
        elif i == 'q':
            score -= 9
        elif i == 'r':
            score -= 5
        elif i == 'n' or i == 'b':
            score -= 3
        elif i == 'p':
            score += 1
    
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
    return [pieces_score(board), int(board.is_check()), nb_wmoves(board), nb_bmoves(board), nb_wattack(board), nb_battack(board), nb_wpin(board), nb_bpin(board)]

board = random_board(['K', 'k', 'N', 'Q', 'q', 'r', 'R'])
print(features(board))