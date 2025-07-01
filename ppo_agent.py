import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pandas as pd
from typing import Tuple, List, Dict

#Sinh ra xác suất (probability distribution) cho các hành động.
class PolicyNetwork(nn.Module):
    #state_dim: số chiều của vector đầu vào (mỗi trạng thái)
    #action_dim: số lượng hành động có thể chọn
    #hidden_dim: số nơ-ron trong mỗi lớp ẩn
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()

        #Biến đổi vector đầu vào state_dim chiều để nhận được xác suất  chọn từng hành động cho action_dim hành động 
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 =  nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

#Dự đoán giá trị của trạng thái (V(s)), dùng để tính advantage và reward.
class ValueNetwork(nn.Module):
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
    def __init__(self, state_dim, action_spaces: Dict[str, List],
                 lr, gamma, eps_clip, k_epochs, c1, c2):
        self.state_dim = state_dim
        self.action_spaces = action_spaces
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.c1 = c1
        self.c2 = c2

        self.action_dim = len(action_spaces['A3_OFFSET']) * len(action_spaces['TTT'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, self.action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.memory = []

    #Chuyển chỉ số hành động (flattened index) thành dict tham số.
    def action_index_to_params(self, action_idx: int) -> Dict[str, int]:
        a3_idx = action_idx // len(self.action_spaces['TTT'])
        ttt_idx = action_idx % len(self.action_spaces['TTT'])
        return {
            'A3_OFFSET': self.action_spaces['A3_OFFSET'][a3_idx],
            'TTT': self.action_spaces['TTT'][ttt_idx]
        }

    #chuyển ngược từ cặp giá trị về index duy nhất trong action space.
    def params_to_action_index(self, a3_offset: int, ttt: int) -> int:
        a3_idx = self.action_spaces['A3_OFFSET'].index(a3_offset)
        ttt_idx = self.action_spaces['TTT'].index(ttt)
        return a3_idx * len(self.action_spaces['TTT']) + ttt_idx

    #Biến đổi Dataframe thành vector đặc trưng.
    def get_state_vector(self, df_window: pd.DataFrame) -> np.ndarray:
        features = []
        if len(df_window) == 0:
            return np.zeros(self.state_dim)

        #chuyển Dataframe thành một vector số đặc trưng gồm các giá trị trung bình, độ lệch chuẩn
        features.extend([
            df_window['RSRP_SOURCE'].mean(),
            df_window['RSRP_SOURCE'].std() if len(df_window) > 1 else 0,
            df_window['RSRP_TARGET'].mean(),
            df_window['RSRP_TARGET'].std() if len(df_window) > 1 else 0,
            df_window['RSRQ_SOURCE'].mean(),
            df_window['RSRQ_SOURCE'].std() if len(df_window) > 1 else 0,
            df_window['RSRQ_TARGET'].mean(),
            df_window['RSRQ_TARGET'].std() if len(df_window) > 1 else 0,
            df_window['HOF'].mean(),
            df_window['PingPong'].mean(),
            df_window['HO_FAIL_COUNT'].mean(),
            len(df_window),
            df_window['LOAD_SOURCE'].mean(),
            df_window['HO_MARGIN'].mean(),
            df_window['TIME_TO_TRIGGER'].mean()
        ])

        #chuẩn hóa độ dài của danh sách feature bằng với state_dim
        features = features[:self.state_dim]
        features.extend([0] * max(0, self.state_dim - len(features)))
        return np.array(features, dtype=np.float32)

    #Dùng policy network sinh phân phối hành động.
    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, float]:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
        dist = Categorical(action_probs)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(action_probs) #Greedy action
        return action.item(), dist.log_prob(action).item()

    #Lưu 1 transition vào bộ nhớ:
    def store_experience(self, state, action, reward, next_state, done, log_prob):
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob
        })

    #Tính reward khi thay đổi action:
    def calculate_reward(self, df_before, df_after, action_params) -> float:
        reward = 0.0

        #Nếu dữ liệu sau hành động (df_after) rỗng, trả về reward -1 (xem như hành động không hiệu quả).
        if len(df_after) == 0:
            return -1.0

        ho_fail = df_after['HOF'].mean()
        ping_pong = df_after['PingPong'].mean()
        rsrp_diff = (df_after['RSRP_TARGET'] - df_after['RSRP_SOURCE']).mean()

        reward -= ho_fail * 10
        reward -= ping_pong * 15

        if rsrp_diff > 0:
            reward += min(rsrp_diff / 10, 2.0)
        else:
            reward -= abs(rsrp_diff) / 5

        #Phạt nhẹ khi A3_OFFSET nằm ngoài khoảng hợp lý (0 hoặc quá lớn ≥10), TTT quá nhỏ (≤40) hoặc quá lớn (≥640).
        if action_params['A3_OFFSET'] == 0 or action_params['A3_OFFSET'] >= 10:
            reward -= 1.0
        if action_params['TTT'] <= 40 or action_params['TTT'] >= 640:
            reward -= 1.0

        #So với df_before, HOF giảm, PingPong giảm thưởng thêm
        if len(df_before) > 0:
            if df_after['HOF'].mean() < df_before['HOF'].mean():
                reward += 5.0
            if df_after['PingPong'].mean() < df_before['PingPong'].mean():
                reward += 5.0

        #Giới hạn reward nhằm tránh chênh lệch quá lớn làm giảm giá trị học
        return np.clip(reward, -20.0, 20.0)

    def update(self):
        if len(self.memory) < 32:
            return

        #Lấy toàn bộ state, action, reward, log_prob từ memory (list of dict), ép thành tensor để tính toán.
        states = torch.tensor(np.array([exp['state'] for exp in self.memory]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([exp['action'] for exp in self.memory]).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in self.memory], dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in self.memory], dtype=torch.float32).to(self.device)

        #Tính phần thưởng cộng dồn có discount VÀ normalize nó
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        #Lặp lại nhiều epoch để update mô hình
        for _ in range(self.k_epochs):
            #Tính lại xác suất hành động
            action_probs = self.policy_net(states) #Dự đoán xác suất hành động từ policy_net
            #Tạo phân phối Categorical để lấy log-prob mới tương ứng với các hành động đã thực hiện
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            #Dự đoán giá trị trạng thái V(s) từ value_net
            values = self.value_net(states).squeeze()
            #Advantage = G_t - V(s): đo lường hành động đó tốt hơn bao nhiêu so với kỳ vọng
            advantages = discounted_rewards - values.detach()

            #Tính tỉ lệ giữa policy mới và cũ
            ratio = torch.exp(new_log_probs - old_log_probs)

            #Tính surr1 và surr2 cho loss PPO (clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()            
            value_loss = nn.MSELoss()(values, discounted_rewards)   #dự đoán V(s) càng gần reward thật càng tốt.
            entropy_loss = -dist.entropy().mean()   #khuyến khích phân phối hành động đa dạng

            total_loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss   #Tổng loss

            #Tính đạo hàm và cập nhật trọng số của cả policy lẫn value network.
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        #Dọn bộ nhớ
        self.memory = []

    #Tính reward cộng dồn có chiết khấu 
    def _calculate_discounted_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        discounted_rewards = torch.zeros_like(rewards)
        running_reward = 0
        for t in reversed(range(len(rewards))):
            running_reward = rewards[t] + self.gamma * running_reward
            discounted_rewards[t] = running_reward
        return discounted_rewards

    #Lưu toàn bộ trạng thái của agent (mạng và optimizer) xuống file để có thể khôi phục sau này
    def save_model(self, path: str):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, path)

    #Tải lại mô hình và trạng thái optimizer từ file đã lưu, giúp tiếp tục training hoặc inference.
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


class HandoverEnvironment:
    def __init__(self, df: pd.DataFrame, window_size: int = 50):
        self.df = df.copy()
        self.window_size = window_size
        self.current_step = 0
        self.reset()

    #Đặt lại môi trường về trạng thái ban đầu (bắt đầu từ đầu dữ liệu).
    def reset(self):
        self.current_step = 0
        return self._get_current_state()

    #Lấy dữ liệu cửa sổ hiện tại quanh current_step
    def _get_current_state(self):
        start = max(0, self.current_step - self.window_size)
        end = min(len(self.df), self.current_step + self.window_size)
        return self.df.iloc[start:end].copy()

    #Thực hiện một bước trong môi trường với các tham số hành động (action_params), cập nhật dữ liệu, chuyển trạng thái
    def step(self, action_params: Dict[str, int]) -> Tuple[pd.DataFrame, bool]:
        current_data = self._get_current_state()
        next_data = current_data.copy()

        if len(next_data) > 0:
            next_data.loc[:, 'HO_MARGIN'] = action_params['A3_OFFSET']
            next_data.loc[:, 'TIME_TO_TRIGGER'] = action_params['TTT']

            ttt_factor = action_params['TTT'] / 320
            a3_factor = action_params['A3_OFFSET'] / 6

            if ttt_factor < 0.5:
                next_data.loc[:, 'PingPong'] |= (np.random.random(len(next_data)) < 0.3)

            if a3_factor > 1.5:
                next_data.loc[:, 'PingPong'] |= (np.random.random(len(next_data)) < 0.2)

        self.current_step += self.window_size // 2
        done = self.current_step >= len(self.df) - self.window_size
        return next_data, done
