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
        self.player = -1 * game_state.current_player

    def is_fully_expanded(self):
        """
        Check if all possible moves from this node have been tried.
        """
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.41):
        """
        Select the child with the highest UCT value.
        """
        max_uct = float('-inf')
        max_index = -1
        for i in range(len(self.children)):
            child = self.children[i]
            if (child.wins / child.visits + c_param * math.sqrt(math.log(self.visits) / child.visits)) > max_uct:
                max_uct = child.wins / child.visits + c_param * math.sqrt(math.log(self.visits) / child.visits)
                max_index = i

        return self.children[max_index]

    def expand(self):
        """
        Expand the node by adding a new child node for one of the untried moves.
        """
        move = self.untried_moves.pop()
        next_state = self.game_state.clone()
        next_state = next_state.make_move(move)
        child_node = Node(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        """
        Backpropagate the result of a simulation up the tree.
        """
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

    def is_terminal_node(self):
        """
        Check if the node is a terminal node (game over).
        """
        return self.game_state.is_terminal()