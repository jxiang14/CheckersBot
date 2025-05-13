import numpy as np
import random
import pickle
from collections import defaultdict

BOARD_SIZE = 8
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        # Q-table keys: (state_key, action_tuple)
        self.q_table = defaultdict(float)

    def get_action(self, state):
        # ε-greedy over valid actions
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_vals = [(self.q_table[(state, action)], action) for action in valid_actions]
        max_q = max(q_vals, key=lambda x: x[0])[0]
        best = [action for q, action in q_vals if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, done):
        current = self.q_table[(state, action)]
        if done:
            target = reward
        else:
            next_actions = self.env.get_valid_actions(next_state)
            next_max = max((self.q_table[(next_state, a)] for a in next_actions), default=0.0)
            target = reward + self.gamma * next_max
        self.q_table[(state, action)] = current + self.lr * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save Q-table and exploration settings to disk."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        """Load Q-table and exploration settings from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(float, data['q_table'])
        self.epsilon = data.get('epsilon', self.epsilon)

class KivyCheckersEnv:
    """
    Adapts CheckersBoard for Q-learning.
    Action: ((r, c), (r2, c2))
    State: tuple of rows with (color, king) or None
    """
    def __init__(self, board_widget):
        self.board = board_widget

    def reset(self):
        self.board.cells.clear()
        self.board.initialize_board()
        self.board.current_turn = "red"
        return self._encode_state()

    def step(self, action):
        (r, c), (r2, c2) = action
        # Clear any highlighting to avoid errors
        self.board.highlight_rect = None
        # Prepare for move
        self.board.selected_position = (r, c)
        self.board.selected_piece = self.board.cells.get((r, c))
        # count pieces before to measure captures
        prev_piece_count = len(self.board.cells)
        try:
            self.board.move_piece(r2, c2)
        except ValueError:
            pass
        # count after move
        new_piece_count = len(self.board.cells)
        captures = prev_piece_count - new_piece_count
        # Reward shaping:
        # - small step penalty to encourage faster play
        # - bonus per capture in this move
        # - larger bonus for winning
        step_penalty = -0.01
        capture_bonus = 1.0 * captures
        done = self.board.check_win_condition()
        win_bonus = 10.0 if done else 0.0
        reward = step_penalty + capture_bonus + win_bonus
        next_state = self._encode_state()
        return next_state, reward, done, {}

    def get_valid_actions(self, state):
        self._decode_state(state)
        actions = []
        for (r, c), (color, king) in self.board.cells.items():
            if color == self.board.current_turn:
                for (r2, c2) in self.board.get_valid_moves(r, c):
                    actions.append(((r, c), (r2, c2)))
        return actions

    def _encode_state(self):
        mat = []
        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                row.append(self.board.cells.get((r, c), None))
            mat.append(tuple(row))
        return (tuple(mat), self.board.current_turn)

    def _decode_state(self, state_key):
        mat, turn = state_key
        self.board.cells.clear()
        for r, row in enumerate(mat):
            for c, cell in enumerate(row):
                if cell:
                    self.board.cells[(r, c)] = cell
        self.board.current_turn = turn

def train_agent(env, agent, episodes=50, max_steps=200):
    for ep in range(1, episodes+1):
        state = env.reset()
        total_reward = 0
        done = False
        for _ in range(max_steps):
            action = agent.get_action(state)
            if action is None:
                break
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        if ep % 1 == 0:
            black_pieces = len([1 for (r, c), (color, _) in env.board.cells.items() if color == "black"])
            red_pieces = len([1 for (r, c), (color, _) in env.board.cells.items() if color == "red"])
            print(f"Episode {ep}/{episodes} Reward={total_reward:.2f} Epsilon={agent.epsilon:.3f} Red={red_pieces} Black={black_pieces}")
    print("Training complete.")

if __name__ == "__main__":
    from board import CheckersBoard
    # Setup board and stub turn_label
    board = CheckersBoard()
     # Your existing Kivy CheckersBoard widget
    from kivy.uix.label import Label
    board.turn_label = Label(text="", color=(0,0,0,1))

    env = KivyCheckersEnv(board)
    agent = QLearningAgent(env)

    # To load an existing model, uncomment:
    # agent.load('qtable.pkl')

    train_agent(env, agent)
    # Save trained Q-table
    agent.save('qtable.pkl')

    # Example: Play against trained agent
    state = env.reset()
    # while True:
    #     # agent move
    #     action = agent.get_action(state)
    #     if action is None: break
    #     state, _, done, _ = env.step(action)
    #     if done: break
    #     # here you could prompt human move via UI/input
    print("Game over.")
