#pip install torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
from typing import Tuple, List, Dict, Any
import random
from collections import deque

class PolicyNetwork(nn.Module):
    """Policy Network cho PPO Agent"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    """Value Network cho PPO Agent"""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class PPOAgent:
    """PPO Agent cho việc tối ưu hóa tham số handover"""
    
    def __init__(self, 
                 state_dim: int,
                 action_spaces: Dict[str, List],
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 c1: float = 0.5,
                 c2: float = 0.01):
        """
        Args:
            state_dim: Kích thước state vector
            action_spaces: Dictionary chứa action spaces cho từng tham số
                          {'A3_OFFSET': [0, 2, 4, 6, 8, 10], 'TTT': [40, 80, 160, 320, 640]}
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs per update
            c1: Value function coefficient
            c2: Entropy coefficient
        """
        self.state_dim = state_dim
        self.action_spaces = action_spaces
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.c1 = c1
        self.c2 = c2
        
        # Tính tổng số action combinations
        self.action_dim = len(action_spaces['A3_OFFSET']) * len(action_spaces['TTT'])
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, self.action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Memory
        self.memory = []
        
    def action_index_to_params(self, action_idx: int) -> Dict[str, int]:
        """Chuyển đổi action index thành parameters"""
        a3_len = len(self.action_spaces['A3_OFFSET'])
        a3_idx = action_idx // len(self.action_spaces['TTT'])
        ttt_idx = action_idx % len(self.action_spaces['TTT'])
        
        return {
            'A3_OFFSET': self.action_spaces['A3_OFFSET'][a3_idx],
            'TTT': self.action_spaces['TTT'][ttt_idx]
        }
    
    def params_to_action_index(self, a3_offset: int, ttt: int) -> int:
        """Chuyển đổi parameters thành action index"""
        a3_idx = self.action_spaces['A3_OFFSET'].index(a3_offset)
        ttt_idx = self.action_spaces['TTT'].index(ttt)
        return a3_idx * len(self.action_spaces['TTT']) + ttt_idx
    
    def get_state_vector(self, df_window: pd.DataFrame) -> np.ndarray:
        """Tạo state vector từ window dữ liệu"""
        if len(df_window) == 0:
            return np.zeros(self.state_dim)
        
        # Tính các features từ data window
        features = []
        
        # RSRP statistics
        features.extend([
            df_window['RSRP_SOURCE'].mean(),
            df_window['RSRP_SOURCE'].std() if len(df_window) > 1 else 0,
            df_window['RSRP_TARGET'].mean(),
            df_window['RSRP_TARGET'].std() if len(df_window) > 1 else 0,
        ])
        
        # RSRQ statistics
        features.extend([
            df_window['RSRQ_SOURCE'].mean(),
            df_window['RSRQ_SOURCE'].std() if len(df_window) > 1 else 0,
            df_window['RSRQ_TARGET'].mean(),
            df_window['RSRQ_TARGET'].std() if len(df_window) > 1 else 0,
        ])
        
        # Handover statistics
        features.extend([
            df_window['HOF'].mean(),  # Failure rate
            df_window['PingPong'].mean(),  # Ping-pong rate
            df_window['HO_FAIL_COUNT'].mean(),
            len(df_window),  # Number of handovers in window
        ])
        
        # Load and margin
        features.extend([
            df_window['LOAD_SOURCE'].mean(),
            df_window['HO_MARGIN'].mean(),
            df_window['TIME_TO_TRIGGER'].mean(),
        ])
        
        # Ensure fixed dimension
        features = features[:self.state_dim]
        features.extend([0] * max(0, self.state_dim - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, float]:
        """Chọn action dựa trên policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            
        if explore:
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        else:
            action = torch.argmax(action_probs)
            action_logprob = torch.log(action_probs.squeeze()[action])
            
        return action.item(), action_logprob.item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool, log_prob: float):
        """Lưu experience vào memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })
    
    def calculate_reward(self, df_before: pd.DataFrame, df_after: pd.DataFrame, 
                        action_params: Dict[str, int]) -> float:
        """Tính reward dựa trên performance metrics"""
        reward = 0.0
        
        if len(df_after) == 0:
            return -1.0  # Penalty for no data
        
        # Calculate metrics after applying action
        ho_fail_rate_after = df_after['HOF'].mean()
        ping_pong_rate_after = df_after['PingPong'].mean()
        avg_rsrp_diff_after = (df_after['RSRP_TARGET'] - df_after['RSRP_SOURCE']).mean()
        
        # Reward components
        # 1. Minimize handover failure rate
        reward -= ho_fail_rate_after * 10
        
        # 2. Minimize ping-pong rate
        reward -= ping_pong_rate_after * 15
        
        # 3. Optimize RSRP difference (should be positive but not too high)
        if avg_rsrp_diff_after > 0:
            reward += min(avg_rsrp_diff_after / 10, 2.0)
        else:
            reward -= abs(avg_rsrp_diff_after) / 5
        
        # 4. Penalty for extreme parameter values
        if action_params['A3_OFFSET'] == 0 or action_params['A3_OFFSET'] >= 10:
            reward -= 1.0
        if action_params['TTT'] <= 40 or action_params['TTT'] >= 640:
            reward -= 1.0
            
        # 5. Bonus for improvement compared to before
        if len(df_before) > 0:
            ho_fail_rate_before = df_before['HOF'].mean()
            ping_pong_rate_before = df_before['PingPong'].mean()
            
            if ho_fail_rate_after < ho_fail_rate_before:
                reward += 5.0
            if ping_pong_rate_after < ping_pong_rate_before:
                reward += 5.0
        
        return np.clip(reward, -20.0, 20.0)
    
    def update(self):
        """Cập nhật policy và value networks sử dụng PPO"""
        if len(self.memory) < 32:  # Minimum batch size
            return
        
        # Convert memory to tensors
        states = torch.FloatTensor([exp['state'] for exp in self.memory]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in self.memory]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in self.memory]).to(self.device)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # PPO update
        for _ in range(self.k_epochs):
            # Get current policy and value predictions
            action_probs = self.policy_net(states)
            values = self.value_net(states).squeeze()
            
            # Calculate policy loss
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate advantages
            advantages = discounted_rewards - values.detach()
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, discounted_rewards)
            
            # Entropy loss
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
        
        # Clear memory
        self.memory = []
    
    def _calculate_discounted_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Tính discounted rewards"""
        discounted_rewards = torch.zeros_like(rewards)
        running_reward = 0
        
        for t in reversed(range(len(rewards))):
            running_reward = rewards[t] + self.gamma * running_reward
            discounted_rewards[t] = running_reward
            
        return discounted_rewards
    
    def save_model(self, filepath: str):
        """Lưu model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

class HandoverEnvironment:
    """Môi trường mô phỏng cho handover optimization"""
    
    def __init__(self, df: pd.DataFrame, window_size: int = 50):
        self.df = df.copy()
        self.window_size = window_size
        self.current_step = 0
        self.reset()
    
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        return self._get_current_state()
    
    def _get_current_state(self) -> pd.DataFrame:
        """Lấy window data hiện tại"""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = min(len(self.df), self.current_step + self.window_size)
        return self.df.iloc[start_idx:end_idx].copy()
    
    def step(self, action_params: Dict[str, int]) -> Tuple[pd.DataFrame, bool]:
        """Thực hiện action và trả về next state"""
        # Simulate applying parameters (trong thực tế sẽ gửi tới Near-RT RIC)
        current_data = self._get_current_state()
        
        # Apply parameters to future handovers (simulation)
        next_data = current_data.copy()
        if len(next_data) > 0:
            # Simulate effect of parameter changes
            next_data.loc[:, 'HO_MARGIN'] = action_params['A3_OFFSET']
            next_data.loc[:, 'TIME_TO_TRIGGER'] = action_params['TTT']
            
            # Simple simulation of parameter effects
            # Lower TTT might increase ping-pong but reduce late handovers
            # Higher A3_OFFSET might reduce failures but increase ping-pong
            ttt_factor = action_params['TTT'] / 320  # Normalize around 320ms
            a3_factor = action_params['A3_OFFSET'] / 6  # Normalize around 6dB
            
            # Adjust failure rate based on parameters
            if ttt_factor < 0.5:  # Very low TTT
                next_data.loc[:, 'PingPong'] = next_data['PingPong'] | (np.random.random(len(next_data)) < 0.3)
            
            if a3_factor > 1.5:  # Very high A3_OFFSET
                next_data.loc[:, 'PingPong'] = next_data['PingPong'] | (np.random.random(len(next_data)) < 0.2)
        
        self.current_step += self.window_size // 2
        done = self.current_step >= len(self.df) - self.window_size
        
        return next_data, done