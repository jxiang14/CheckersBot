
# def minimaxAction(board, depth, player_color):
#     """
#     Returns the best action for the player using the minimax algorithm with alpha-beta pruning.
#     """
#     alpha = float('-inf')
#     beta = float('inf')

#     best_action = minimax(board, depth, player_color, alpha, beta)

#     return best_action

# def minimax(board, depth, player_color, alpha, beta):
#     """
#     Minimax algorithm with alpha-beta pruning.
#     """
#     # if depth == 0 or board.check_win_condition():
#     #     return board.evaluate(player_color)

#     # current_player = board.get_current_player()

#     # best_action = None

#     # if current_player == player_color:
#     #     max_val = float('-inf')
#     #     for action in board.get_all_moves(current_player):
#     #         new_board = board.apply_move(action)
#     #         value = minimax(new_board, depth - 1, player_color, alpha, beta)
#     #         if value > max_val:
#     #             max_val = value
#     #             best_action = action
#     #         alpha = max(alpha, value)
#     #         if beta <= alpha:
#     #             break
#     # else:
#     #     min_val = float('inf')
#     #     for action in board.get_all_moves(current_player):
#     #         new_board = board.apply_move(action)
#     #         value = minimax(new_board, depth - 1, player_color, alpha, beta)
#     #         if value < min_val:
#     #             min_val = value
#     #             best_action = action
#     #         beta = min(beta, value)
#     #         if beta <= alpha:
#     #             break

#     # return best_action

#     if depth == 0 or board.check_win_condition():
#             return None, board.evaluate(player_color)

#     current_player = board.get_current_player()
#     all_moves = board.get_all_moves(current_player)

#     best_action = None
    
#     if current_player == player_color:
#         max_eval = float('-inf')
#         for action in all_moves:
#             new_board = board.apply_move(action)
#             _, eval = minimax(new_board, depth - 1, player_color, alpha, beta)
#             if eval > max_eval:
#                 max_eval = eval
#                 best_action = action
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break
#         return best_action, max_eval
#     else:
#         min_eval = float('inf')
#         for action in all_moves:
#             new_board = board.apply_move(action)
#             _, eval = minimax(new_board, depth - 1, player_color, alpha, beta)
#             if eval < min_eval:
#                 min_eval = eval
#                 best_action = action
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 break
#         return best_action, min_eval


import copy
from utils import CheckersState

def get_best_move(board, player_color, depth=3):
    
    if player_color == "black":
        player_color = "red"
    else:
        player_color = "black"
    board = CheckersState(player_color, board)
    best_action, _ = minimaxAction(board, depth, player_color, float('-inf'), float('inf'), True)
    # print("[DEBUG] Best action found:", best_action)
    if best_action is None:
        # print("[DEBUG] No best action found, returning None")
        return (None, None)
    return best_action

def minimaxAction(board:CheckersState, depth, player_color, alpha, beta, maximizing_player):
    best_move = None

    if depth == 0 or board.get_winner():
        eval_score = evaluate(board, player_color)
        return None, eval_score
    
    current_color = player_color if maximizing_player else get_opponent(player_color)
    all_moves = get_all_moves(board, current_color)

    if maximizing_player:
        max_eval = float('-inf')
        for from_pos, to_pos in all_moves:
            new_board = simulate_move(board, from_pos, to_pos, player_color)
            _, eval = minimaxAction(new_board, depth - 1, player_color, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = (from_pos, to_pos)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return best_move, max_eval
    else:
        min_eval = float('inf')
        for from_pos, to_pos in all_moves:
            new_board = simulate_move(board, from_pos, to_pos, player_color)
            _, eval = minimaxAction(new_board, depth - 1, player_color, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = (from_pos, to_pos)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return best_move, min_eval


def get_opponent(color):
    return "black" if color == "red" else "red"

def get_all_moves(board, color):
    # moves = []
    # for (row, col), (piece_color, _) in board.cells.items():
    #     if piece_color != color:
    #         continue
    #     valid_moves = board.get_valid_moves(row, col)
    #     for move in valid_moves:
    #         moves.append(((row, col), move))
    # return moves
    return board.get_all_valid_moves(color)


def simulate_move(board, from_pos, to_pos, player_color):
    new_board = board.clone()
    new_board.make_move((from_pos, to_pos))
    return new_board


def evaluate(board, player_color):
    score = 0

    for row in range(8):
        for column in range(8):
            piece = board.board[row][column]
            if piece == 0:
                continue
            color, is_king = piece
            if color == player_color:
                score += 1
                if is_king:
                    score += 1
            else:
                score -= 1
                if is_king:
                    score -= 1
    return score
