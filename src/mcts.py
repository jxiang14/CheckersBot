from mcts_node import Node
import random

RED = 1
BLACK = -1

class MCTS:
    def __init__(self, game_state, moves_made, iteration_limit=1500):
        self.root = Node(game_state)
        self.iteration_limit = iteration_limit
        self.epsilon = 1.0
        self.moves_made = moves_made

    def run(self):
        """
        Run the MCTS algorithm for a specified number of iterations.
        """
        sim_avg = 0
        for _ in range(self.iteration_limit):
            node = self._select(self.root)
            result = self._simulate(node.game_state)
            sim_avg += result
            node.backpropagate(result)
        print(f"Average simulation result: {sim_avg / self.iteration_limit:.2f}")

        return self._best_move()

    def _select(self, node):
        """
        Select a node to expand using the UCT algorithm.
        """
        while not node.is_terminal_node():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def _simulate(self, game_state):
        """
        Simulate a random game from the given state until a terminal state is reached
        or a maximum game depth is reached.

        args:
            game_state: The current game state to simulate from.
        returns:
            The result of the simulation: 1 for a win, -1 for a loss, and 0 for a draw.
        """
        state = game_state.clone()
        iterations = 0
        if self.moves_made > 150:
            max_depth = 40
        else:
            max_depth = 200
        while not state.is_terminal() and iterations < max_depth:
            moves = state.get_all_valid_moves(state.current_player)
            if not moves:
                break
            move = random.choice(moves)
            state = state.make_move(move)
            iterations += 1

        root_player = self.root.game_state.current_player
        winner = state.get_winner()
        if winner == root_player:
            return 1
        elif winner == 0:
            return 0
        else:
            return -1

    def _best_move(self):
        """
        Returns the best move based on the number of visits to each child node.
        """
        if not self.root.children:
            return None
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.move