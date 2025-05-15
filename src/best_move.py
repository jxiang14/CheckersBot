import minimax
from utils import CheckersState
from mcts import MCTS

# def get_best_move(board, player_color):
#     """
#     Get the best move for the player using the minimax algorithm with alpha-beta pruning.
#     """
#     best_action = minimax.get_best_move(board, player_color, depth=3)

#     if best_action is None:
#         return (None, None)
    
#     return best_action

def get_best_move(board, player_color):
    game_state = CheckersState(player_color, board)
    mcts = MCTS(game_state, iteration_limit=2500)
    best_move = mcts.run()
    return best_move[0], best_move[1]
