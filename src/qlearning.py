import random
import pickle
from collections import defaultdict
from utils import CheckersState, RED, BLACK
BOARD_SIZE = 8

class QLearningAgent:
    def __init__(self, alpha=0.01, gamma=0.9, epsilon=0.1, epsilon_decay=0.999):
        self.alpha = alpha  # learning rate for weight updates
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # decay per episode
        self.weights = defaultdict(float)

    def get_features(self, state, action):
        next_state = state.clone()
        next_state.make_move(action)

        def count(board, color, king_flag=None):
            return sum(
                1
                for row in board
                for cell in row
                if cell != 0 and cell[0] == color and (king_flag is None or cell[1] == king_flag)
            )

        my_color = state.current_player
        opp_color = -my_color
        my_pieces_before = count(state.board, my_color)
        opp_pieces_before = count(state.board, opp_color)
        my_pieces_after = count(next_state.board, my_color)
        opp_pieces_after = count(next_state.board, opp_color)
        my_kings_before = count(state.board, my_color, king_flag=True)
        opp_kings_before = count(state.board, opp_color, king_flag=True)
        my_kings_after = count(next_state.board, my_color, king_flag=True)
        opp_kings_after = count(next_state.board, opp_color, king_flag=True)

        features = {
            'bias': 1.0,
            'piece_diff': (my_pieces_after - opp_pieces_after) - (my_pieces_before - opp_pieces_before),
            'king_diff': (my_kings_after - opp_kings_after) - (my_kings_before - opp_kings_before),
            'capture': 1.0 if opp_pieces_after < opp_pieces_before else 0.0,
            'king_capture': 1.0 if opp_kings_after < opp_kings_before else 0.0,
            'promotion': 1.0 if my_kings_after > my_kings_before else 0.0,
        }
        (r0, c0), (r1, c1) = action
        delta = (r1 - r0) * (1 if my_color == 1 else -1)
        features['forward'] = max(delta, 0) / BOARD_SIZE
        return features

    def q_value(self, state, action):
        feats = self.get_features(state, action)
        return sum(self.weights[f] * v for f, v in feats.items())

    def choose_action(self, state):
        moves = state.get_all_valid_moves(state.current_player)
        if not moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(moves)
        q_vals = [(self.q_value(state, a), a) for a in moves]
        max_q = max(q_vals, key=lambda x: x[0])[0]
        best = [a for q, a in q_vals if q == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state):
        next_moves = next_state.get_all_valid_moves(next_state.current_player)
        max_next = max((self.q_value(next_state, a) for a in next_moves), default=0.0)
        target = reward + self.gamma * max_next
        prediction = self.q_value(state, action)
        error = target - prediction
        feats = self.get_features(state, action)
        for f, v in feats.items():
            self.weights[f] += self.alpha * error * v

    def run_episode(self, training=True):
        from random import choice
        state = CheckersState('red')
        total_reward = 0.0
        # Alternate turns, both using agent
        while not state.is_terminal():
            action = self.choose_action(state)
            if action is None:
                break
            next_state = state.clone()
            next_state.make_move(action)
            # reward for the acting player
            if next_state.is_terminal():
                winner = next_state.get_winner()
                reward = 1.0 if winner == state.current_player else -1.0
            else:
                feats = self.get_features(state, action)
                reward = (
                    feats['king_capture'] * 3.0 +
                    feats['capture'] * 2.0 +
                    feats['promotion'] * 1.0 +
                    feats['forward'] * 0.5
                )
            if training:
                self.update(state, action, reward, next_state)
            # accumulate reward only for one perspective (e.g., RED)
            total_reward += reward if state.current_player == RED else -reward
            state = next_state
            state.switch_player()
        return total_reward

    def train(self, episodes=10000, log_interval=1000):
        for ep in range(1, episodes + 1):
            self.run_episode(training=True)
            self.epsilon *= self.epsilon_decay
            if ep % log_interval == 0:
                print(f"Episode {ep}")
                self.evaluate(episodes=log_interval)
        print("Training complete")

    def evaluate(self, episodes=1000):
        total = 0.0
        old_eps = self.epsilon
        self.epsilon = 0.0
        for _ in range(episodes):
            total += self.run_episode(training=False)
        self.epsilon = old_eps
        avg = total / episodes
        print(f"Avg reward: {avg}")
        return avg

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.weights), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.weights = defaultdict(float, pickle.load(f))

if __name__ == "__main__":
    agent = QLearningAgent(alpha=0.01, gamma=0.95, epsilon=0.1)
    agent.train(episodes=100000)
    agent.save('checkers_weights.pkl')
