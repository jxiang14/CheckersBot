import copy
from utils import CheckersState

RED = 1
BLACK = -1

def get_best_move(board, player_color, depth=3):
    board = CheckersState(player_color, board)
    best_action, _ = minimaxAction(board, depth, player_color, float('-inf'), float('inf'), True)
    # print("[DEBUG] Best action found:", best_action)
    if best_action is None:
        # print("[DEBUG] No best action found, returning None")
        return (None, None)
    return best_action

def minimaxAction(state:CheckersState, depth, player_color, alpha, beta, maximizing_player):
    best_move = None

    if depth == 0 or state.get_winner():
        eval_score = evaluate(state, player_color)
        return None, eval_score
    
    current_color = state.current_player
    all_moves = get_all_moves(state, current_color)

    if maximizing_player:
        max_eval = float('-inf')
        for move in all_moves:
            new_state = simulate_move(state, move)
            
            #new code
            if depth == 1:
                # print("depth is currently ", depth)
                eval = evaluate_move(new_state, move, player_color)
                print(f"Evaluating move {move} with eval_move: {eval}")
            else:
                # print("depth is currently ", depth)
                _, eval = minimaxAction(new_state, depth - 1, player_color, alpha, beta, False)
                print(f"Evaluating move {move} with minimax: {eval}")
            
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return best_move, max_eval
    else:
        min_eval = float('inf')
        for move in all_moves:
            new_state = simulate_move(state, move)
            
            #new code
            if depth == 1:
                eval = evaluate_move(new_state, move, player_color)
                # print(f"Evaluating move {from_pos} -> {to_pos} with eval_move: {eval}")
            else:
                _, eval = minimaxAction(new_state, depth - 1, player_color, alpha, beta, True)
                # print(f"Evaluating move {from_pos} -> {to_pos} with minimax: {eval}")
            
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return best_move, min_eval


def get_opponent(color):
    return BLACK if color == RED else RED

def get_all_moves(state, color):
    # moves = []
    # for (row, col), (piece_color, _) in state.cells.items():
    #     if piece_color != color:
    #         continue
    #     valid_moves = state.get_valid_moves(row, col)
    #     for move in valid_moves:
    #         moves.append(((row, col), move))
    # return moves
    return state.get_all_valid_moves(color)


def simulate_move(state, move):
    new_state = state.clone()
    new_state.make_move(move)
    return new_state


def evaluate(state, player_color):
    score = 0

    if player_color == "red":
        player_color = RED
    else:
        player_color = BLACK

    for row in range(8):
        for column in range(8):
            piece = state.board[row][column]
            if piece == 0:
                continue
            color, is_king = piece
            piece_score = 0
            
            if is_king:
                piece_score += 2.0
            else:
                piece_score += 1.0
                if color == RED:
                    piece_score += row * 0.05 # more points to a red piece that is closer to being a king
                else:
                    piece_score += (7 - row) * 0.05 

            if color == player_color:
                score += piece_score
            else:
                score -= piece_score
                
    return score

def evaluate_move(state, move, player_color):
    score = 0
    if len(move) == 2:
        if state.can_capture_single_piece_if_moved(move):
            score += 2
    if len(move) > 2:
        score += 5
    if state.can_become_king(move):
        score += 3
    if state.can_be_captured_if_moved(move):
        score -= 3
    if state.can_be_at_edge(move):
        score += 0.5
    score += evaluate(state, player_color)
    print("player color: ", player_color)
    # POSSIBLE HEURISTICS 
    # add backed up (more safe)
    # add new exposures (less safe)
    # add trap opponent
    # print (f"MOVE: {move}, SCORE: {score}")
    return score
