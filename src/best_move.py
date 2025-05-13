import minimax

def get_best_move(board, player_color):
    """
    Get the best move for the player using the minimax algorithm with alpha-beta pruning.
    """
    best_action = minimax.get_best_move(board, player_color, depth=3)

    if best_action is None:
        return (None, None)
    
    return best_action
