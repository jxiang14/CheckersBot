import random
import copy
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import CheckersState, RED, BLACK

gamma = 0.99
learning_rate = 1e-4
batch_size = 64
buffer_capacity = 100000
update_target_every = 1000

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ReplayBuffer:
    def __init__(self, capacity=buffer_capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_action, reward, next_state_action, done):
        self.buffer.append((state_action, reward, next_state_action, done))

    def sample(self, batch_size=batch_size):
        batch = random.sample(self.buffer, batch_size)
        sa, rewards, next_sa, dones = zip(*batch)
        return (
            torch.stack(sa).to(device),
            torch.tensor(rewards, dtype=torch.float, device=device),
            torch.stack(next_sa).to(device),
            torch.tensor(dones, dtype=torch.float, device=device)
        )

    def __len__(self):
        return len(self.buffer)

class QLearningNetwork(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512):
        super(QLearningNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        return self.fc(flat).squeeze(-1)

class DeepQNetwork(nn.Module):
    def __init__(self, state_encoder, action_encoder,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=1e-5):
        self.q_net = QLearningNetwork().to(device)
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.encode_state = state_encoder   
        self.encode_action = action_encoder
        self.train_steps = 0

    def select_action(self, state, valid_moves, exploit=False):
        sa_tensors = []
        for move in valid_moves:
            s_t = self.encode_state(state)
            a_t = self.encode_action(state, move)
            sa_tensors.append(torch.cat([s_t, a_t], dim=0))
        batch = torch.stack(sa_tensors).to(device)

        if not exploit and random.random() < self.epsilon:
            idx = random.randrange(len(valid_moves))
        else:
            with torch.no_grad():
                q_vals = self.q_net(batch)
            idx = q_vals.argmax().item()

        # decay epsilon
        if not exploit:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return valid_moves[idx], batch[idx]

    def update(self):
        if len(self.replay_buffer) < batch_size:
            return
        sa_batch, rewards, next_sa_batch, dones = self.replay_buffer.sample()

        current_q = self.q_net(sa_batch)
        next_qs = []
        for next_sa in next_sa_batch:
            next_qs.append(self.target_net(next_sa.unsqueeze(0)).item())
        next_q = torch.tensor(next_qs, device=device)
        discount = (1 - dones) * gamma * next_q
        target_q = rewards + discount
      
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt.get('epsilon', self.epsilon)
        self.train_steps = ckpt.get('train_steps', self.train_steps)

    def best_move(self, state):
        """
        Returns the highest-Q move for the current state (no exploration).
        """
        valid_moves = state.get_all_valid_moves(state.current_player)
        best, _ = self.select_action(state, valid_moves, exploit=True)
        return best

def state_to_tensor(state):
    arr = np.zeros((3,8,8), np.float32)
    for r in range(8):
        for c in range(8):
            cell = state.board[r][c]
            if cell != 0:
                color, king = cell
                arr[0 if color == RED else 1, r, c] = 1
                if king:
                    arr[2, r, c] = 1
    return torch.from_numpy(arr)

def action_to_tensor(state, move):
    arr = np.zeros((3,8,8), np.float32)
    for (r, c) in move:
        arr[0, r, c] = 1
    return torch.from_numpy(arr)

def train(agent, num_episodes=10000):
    for ep in range(num_episodes):
        state = CheckersState("red")
        while not state.is_terminal():
            moves = state.get_all_valid_moves(state.current_player)
            move, sa_tensor = agent.select_action(state, moves)
            state.make_move(move)
            if any(abs(e[0]-s[0])==2 for s,e in zip(move[:-1], move[1:])):
                reward = 1
            else:
                reward = 0
            
            if state.is_terminal():
                reward = 10 if state.get_winner() == RED else -10
            next_sa_tensor = torch.cat([
                state_to_tensor(state), action_to_tensor(state, move)], dim=0)
            done = state.is_terminal()
            agent.replay_buffer.push(sa_tensor, reward, next_sa_tensor, done)
            agent.update()

        print(f"Episode {ep}, epsilon {agent.epsilon:.3f}")

if __name__ == "__main__":
    agent = DeepQNetwork(state_to_tensor, action_to_tensor)
    train(agent)
    agent.save("checkers_dqn.pth")