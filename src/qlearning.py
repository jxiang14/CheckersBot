import random
import pickle
from collections import defaultdict
from checkers_state import CheckersState, RED, BLACK

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        # Q-table: mapping state_key -> dict of action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))

    def state_to_key(self, state):
        # Serialize board and current player into a hashable key
        flat = []
        for row in state.board:
            for cell in row:
                if cell == 0:
                    flat.append(0)
                else:
                    color, king = cell
                    val = color * (2 if king else 1)
                    flat.append(val)
        flat.append(state.current_player)
        return tuple(flat)

    def choose_action(self, state):
        moves = state.get_all_valid_moves(state.current_player)
        if not moves:
            return None
        key = self.state_to_key(state)
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(moves)
        # choose max-Q move
        q_vals = self.q_table[key]
        if not q_vals:
            return random.choice(moves)
        best_val = max(q_vals[a] for a in moves)
        best_moves = [a for a in moves if q_vals[a] == best_val]
        return random.choice(best_moves)

    def get_reward(self, state, next_state):
        """
        Reward priorities:
        1. Win: +1, Lose: -1
        2. Capture opponent king: +0.5
        3. Capture any opponent piece: +0.2
        4. Move forward (for RED): +0.05 per row
        """
        # 1. Terminal reward
        if next_state.is_terminal():
            winner = next_state.get_winner()
            return 1.0 if winner == RED else -1.0

        reward = 0.0
        # Count opponent kings before and after
        opponent = BLACK
        # state.board cells hold (color, king)
        def count_kings(board, color):
            return sum(1 for row in board for cell in row if cell != 0 and cell[0] == color and cell[1])
        kings_before = count_kings(state.board, opponent)
        kings_after = count_kings(next_state.board, opponent)
        if kings_after < kings_before:
            return 0.5

        # Count total opponent pieces before and after
        def count_pieces(board, color):
            return sum(1 for row in board for cell in row if cell != 0 and cell[0] == color)
        pieces_before = count_pieces(state.board, opponent)
        pieces_after = count_pieces(next_state.board, opponent)
        if pieces_after < pieces_before:
            return 0.2

        # Motivate forward movement: RED moves down (+row), BLACK moves up (-row)
        # Find moved piece location
        # we assume only one move occurred
        # get positions of agent's pieces
        agent_color = RED
        before_positions = [(r, c) for r in range(len(state.board)) for c in range(len(state.board)) \
                            if state.board[r][c] != 0 and state.board[r][c][0] == agent_color]
        after_positions = [(r, c) for r in range(len(next_state.board)) for c in range(len(next_state.board)) \
                           if next_state.board[r][c] != 0 and next_state.board[r][c][0] == agent_color]
        # find position that changed
        moved = set(after_positions) - set(before_positions)
        if moved:
            new_row, _ = moved.pop()
            # find old position of that piece as the one not in after but in before
            old = set(before_positions) - set(after_positions)
            if old:
                old_row, _ = old.pop()
                delta = new_row - old_row
                # for RED, delta>0 is forward; for BLACK, delta<0
                reward += 0.05 * (delta if agent_color == RED else -delta)
        return reward

    def update(self, state, action, reward, next_state):
        s_key = self.state_to_key(state)
        ns_key = self.state_to_key(next_state)
        q_sa = self.q_table[s_key][action]
        next_moves = next_state.get_all_valid_moves(next_state.current_player)
        max_q_next = max((self.q_table[ns_key][a] for a in next_moves), default=0.0)
        self.q_table[s_key][action] = q_sa + self.alpha * (reward + self.gamma * max_q_next - q_sa)

    def run_episode(self, training=True):
        from random import choice
        state = CheckersState('red')
        total_reward = 0
        while not state.is_terminal():
            if state.current_player == RED:
                action = self.choose_action(state)
                if action is None:
                    break
                next_state = state.clone()
                next_state.make_move(action)
                reward = self.get_reward(state, next_state)
                if training:
                    self.update(state, action, reward, next_state)
                total_reward += reward
            else:
                moves = state.get_all_valid_moves(state.current_player)
                if not moves:
                    break
                action = choice(moves)
                next_state = state.clone()
                next_state.make_move(action)
            state = next_state
            state.switch_player()
        return total_reward

    def train(self, episodes=10000, log_interval=1000):
        for ep in range(1, episodes + 1):
            self.run_episode(training=True)
            if ep % log_interval == 0:
                print(f"Episode {ep}: Q-table size: {len(self.q_table)}")
        print(f"Training completed over {episodes} episodes.")

    def evaluate(self, episodes=1000):
        total = 0
        old_eps = self.epsilon
        self.epsilon = 0.0
        for _ in range(episodes):
            total += self.run_episode(training=False)
        self.epsilon = old_eps
        avg_reward = total / episodes
        print(f"Evaluation over {episodes} episodes: avg reward = {avg_reward}")
        return avg_reward

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))

if __name__ == "__main__":
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.1)
    agent.train(episodes=10000)
    agent.save('checkers_qtable.pkl')
