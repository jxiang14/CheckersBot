import minimax
from checkers_state import CheckersState
from mcts import MCTS

# def get_best_move(board, player_color):
#     """
#     Get the best move for the player using the minimax algorithm with alpha-beta pruning.
#     """
#     best_action = minimax.get_best_move(board, player_color, depth=3)

#     if best_action is None:
#         return (None, None)
    
#     return best_action

def get_best_move(board, player_color, moves_made):
    game_state = CheckersState(player_color, board)
    mcts = MCTS(game_state, moves_made, iteration_limit=1000)
    best_move = mcts.run()
    return best_move
