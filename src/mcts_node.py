import math

class Node:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = game_state.get_all_valid_moves(game_state.current_player)
        self.player = parent.player * -1 if parent else -1 * game_state.current_player

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.41):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.game_state.clone()
        next_state = next_state.make_move(move)
        child_node = Node(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        if result == self.player:
            self.wins += 1
        # print(f"Backpropagating: {self.move}, Color: {self.player} Result: {result}, Wins: {self.wins}, Visits: {self.visits}")
        if self.parent:
            self.parent.backpropagate(result)

    def is_terminal_node(self):
        return self.game_state.is_terminal()