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


print(random_board(['K', 'k', 'N', 'b']))