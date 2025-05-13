

def get_best_move(board, player_color):
    """
    Get the best move for the player using the minimax algorithm.
    """

    for (row, col) in board.cells.keys():
        color = board.cells[(row, col)][0]
        if color == player_color:
            valid_moves = board.get_valid_moves(row, col)
            print(f"Valid moves for {color} at ({row}, {col}): {valid_moves}")
            if len(valid_moves) > 0:
                return (row, col), (valid_moves[0][0], valid_moves[0][1])