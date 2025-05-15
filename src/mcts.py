from mcts_node import Node
import random

RED = 1
BLACK = -1

class MCTS:
    def __init__(self, game_state, iteration_limit=1500):
        self.root = Node(game_state)
        self.iteration_limit = iteration_limit
        self.epsilon = 0.7

    def run(self):
        sim_avg = 0
        for _ in range(self.iteration_limit):
            node = self._select(self.root)
            result = self._simulate(node.game_state)
            sim_avg += result
            node.backpropagate(result)
        print(f"Average simulation result: {sim_avg / self.iteration_limit:.2f}")

        return self._best_move()

    def _select(self, node):
        while not node.is_terminal_node():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def _simulate(self, game_state):
        state = game_state.clone()
        while not state.is_terminal():
            moves = state.get_all_valid_moves(state.current_player)
            if not moves:
                break
            if random.random() < self.epsilon:
                move = random.choice(moves)
            else:
                move = self._evaluate_moves_heuristically(moves, state)
            state = state.make_move(move)
            state.switch_player()

        winner = state.get_winner()
        # print(f"Root player: {self.root.game_state.current_player}")
        # # Return 1 if root's player won, -1 if they lost, 0 for draw
        # root_player = self.root.game_state.current_player
        # return 1 if winner == root_player else -1
        return winner

    def _best_move(self):
        if not self.root.children:
            return None  # No moves available
        for child in self.root.children:
            print(f"Move: {child.move}, Wins: {child.wins}, Visits: {child.visits}, Win rate: {child.wins / child.visits:.2f}")
        # best_child = max(self.root.children, key=lambda c: c.visits)
        best_child = max(self.root.children, key=lambda c: (c.wins / c.visits) + 0.1 * c.visits)
        return best_child.move
    
    def _evaluate_moves_heuristically(self, moves, state):
        best_move = random.choice(moves)
        best_score = -float('inf')
        for move in moves:
            score = 0
            (r1, c1), (r2, c2) = move
            if abs(r2 - r1) > 1:
                score += 10 # capture bonus
            if (state.current_player == RED and r2 == 7) or (state.current_player == BLACK and r2 == 0):
                score += 5  # king promotion
            if self.can_be_captured_if_moved(move, state):
                score -= 10 # penalty for being captured
            if r2 in (0, 7) or c2 in (0, 7):
                score += 2  # edge bonus
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    
    def can_be_captured_if_moved(self, move, state):
        (r1, c1), (r2, c2) = move
        if state.current_player == RED:
            opponent_pieces = state.black_pieces
            dirs = [(-1, -1), (-1, 1)]
        else:
            opponent_pieces = state.red_pieces
            dirs = [(1, -1), (1, 1)]
        for dr, dc in dirs:
            r, c = r2 - dr, c2 - dc
            if (r, c) in opponent_pieces:
                if r2 + dr >= 0 and r2 + dr < 8 and c2 + dc >= 0 and c2 + dc < 8:
                    if c2 + dc == c1 or state.board[r2 + dr][c2 + dc] == 0:
                        return True
        return False