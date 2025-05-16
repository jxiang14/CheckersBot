import random
import pickle
import numpy as np
from collections import defaultdict
from utils import CheckersState

class QAgent:
    def __init__(
        self,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9999,
        epsilon_min=0.1,
        q_table_path=None
    ):
        # Q-table: key = (state_repr, action_repr), value = Q-value
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Optionally load an existing Q-table
        if q_table_path:
            self.load_q_table(q_table_path)

    def choose_action(self, state_repr, legal_actions):
        if not legal_actions:
            return None
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        q_values = []
        for actions in legal_actions:
            value = self.Q[(state_repr, self._action_to_key(actions))]
            q_values+= value
        index = np.argmax(q_values)
        max_idx = int(index)
        return legal_actions[max_idx]

    def update(self, s_repr, action, reward, next_s_repr, next_legal_actions):
        action_key = self._action_to_key(action)
        current_q = self.Q[(s_repr, action_key)]
        next_qs = []
        if next_legal_actions:
            for action in next_legal_actions:
                action_key = self._action_to_key(action)
                next_qs += [self.Q[(next_s_repr, action_key)]]
            max_next_q = max(next_qs)
        else:
            max_next_q = 0.0
        target = reward + self.gamma * max_next_q
        self.Q[(s_repr, action_key)] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, path='q_table.pkl'):
        """Save the Q-table to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def load_q_table(self, path):
        """Load the Q-table from a pickle file."""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.Q = defaultdict(float, data)
            print(f"Loaded Q-table from {path}, entries: {len(self.Q)}")
        except FileNotFoundError:
            print(f"Q-table file {path} not found. Starting with empty Q-table.")

    @staticmethod
    def _state_to_key(state: CheckersState):
        flat = []
        for row in state.board:
            for cell in row:
                if cell == 0:
                    flat.append(0)
                else:
                    color, king = cell
                    flat.append(color * (2 if king else 1))
        return tuple(flat) + (state.current_player,)

    @staticmethod
    def _action_to_key(action):
        return tuple(tuple(step) for step in action)


def train(agent: QAgent, episodes: int = 50000, save_path='q_table.pkl'):
    for ep in range(episodes):
        state = CheckersState('red')
        done = False
        while not done:
            s_key = QAgent._state_to_key(state)
            legal = state.get_all_valid_moves(state.current_player)
            action = agent.choose_action(s_key, legal)
            if action is None:
                break
            next_state = state.clone().make_move(action)
            reward = 0
            done = next_state.is_terminal()
            if done:
                winner = next_state.get_winner()
                reward = 1 if winner == state.current_player else -1
            next_key = QAgent._state_to_key(next_state)
            next_legal = next_state.get_all_valid_moves(next_state.current_player)
            agent.update(s_key, action, reward, next_key, next_legal)
            state = next_state
        agent.decay_epsilon()
        if (ep + 1) % 1000 == 0:
            print(f"Episode {ep+1}/{episodes}, epsilon={agent.epsilon:.4f}")
    agent.save_q_table(save_path)
    print(f"Training complete. Q-table saved to {save_path}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train or load Q-learning agent for checkers')
    parser.add_argument('--load', type=str, default=None, help='Path to load existing Q-table')
    parser.add_argument('--episodes', type=int, default=100000, help='Number of training episodes')
    parser.add_argument('--save', type=str, default='q_table.pkl', help='Path to save Q-table')
    args = parser.parse_args()

    agent = QAgent(q_table_path=args.load)
    train(agent, episodes=args.episodes, save_path=args.save)
