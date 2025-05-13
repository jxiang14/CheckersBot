
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

def get_best_move(board, player_color, depth=3):
    best_action, _ = minimaxAction(board, depth, player_color, float('-inf'), float('inf'), True)
    print("[DEBUG] Best action found:", best_action)
    if best_action is None:
        print("[DEBUG] No best action found, returning None")
        return (None, None)
    return best_action


def minimaxAction(board, depth, player_color, alpha, beta, maximizing_player):
    if depth == 0 or board.check_win_condition():
        return None, evaluate(board, player_color)

    current_player = board.current_turn
    legal_actions = get_all_legal_actions(board, current_player)
    print("[DEBUG] Current player:", current_player)
    # print("[DEBUG] Legal actions:", legal_actions)

    if not legal_actions:
        print("[DEBUG] No legal actions available for", current_player)
        return None, evaluate(board, player_color)

    # best_action = None

    if maximizing_player:
        max_eval = float('-inf')
        for action in legal_actions:
            new_board = board.apply_action(action)
            _, eval = minimaxAction(new_board, depth - 1, player_color, alpha, beta, False)
            print("[DEBUG] Evaluation:", eval)
            if eval > max_eval:
                max_eval = eval
                best_action = action
                # print("[DEBUG] Best action for maximizing player:", best_action)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        print("[DEBUG] Best action for maximizing player OUTSIDE LOOP:", best_action)
        return best_action, max_eval
    else:
        min_eval = float('inf')
        for action in legal_actions:
            new_board = board.apply_action(action)
            _, eval = minimaxAction(new_board, depth - 1, player_color, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_action = action
                # print("[DEBUG] Best action for minimizing player:", best_action)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        print("[DEBUG] Best action for minimizing player OUTSIDE LOOP:", best_action)
        return best_action, min_eval



def get_all_legal_actions(board, player_color):
    actions = []
    for (row, col), (color, _) in board.cells.items():
        if color == player_color:
            moves = board.get_valid_moves(row, col)
            for move in moves:
                actions.append(((row, col), move))
    return actions

def evaluate(board, player_color):
    score = 0
    for (color, is_king) in board.cells.values():
        value = 1.5 if is_king else 1
        if color == player_color:
            score += value
        else:
            score -= value
    return score
