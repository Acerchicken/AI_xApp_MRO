
from parse import parse_log_file
from detect_pingpong import detect_ping_pong
from ppo_agent import PPOAgent, HandoverEnvironment
import pandas as pd
import os

def evaluate_performance(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {'handover_failure_rate': 0, 'ping_pong_rate': 0}
    return {
        'handover_failure_rate': df['HOF'].mean(),
        'ping_pong_rate': df['PingPong'].mean()
    }

def train_ppo_agent(df: pd.DataFrame, action_spaces: dict, ppo_config: dict, episodes: int = 10):
    #Tạo env với cửa sổ trượt window_size gồm n dòng
    env = HandoverEnvironment(df, window_size=50)
    agent = PPOAgent(state_dim=15, action_spaces=action_spaces, **ppo_config)
    best_reward = float('-inf')

    for episode in range(episodes):
        #Reset environment và lấy state đầu:
        current_data = env.reset()
        state = agent.get_state_vector(current_data)
        total_reward = 0
        step_count = 0
        episode_experiences = []

        while True:
            #Chọn hành động (tham số A3, TTT):
            action_idx, log_prob = agent.select_action(state, explore=True)
            action_params = agent.action_index_to_params(action_idx)

            #Thực hiện action và nhận phản hồi từ môi trường:
            prev_data = current_data.copy()
            next_data, done = env.step(action_params)
            next_state = agent.get_state_vector(next_data)
            reward = agent.calculate_reward(prev_data, next_data, action_params)
            total_reward += reward

            #Lưu lại kinh nghiệm
            episode_experiences.append({
                'state': state,
                'action': action_idx,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob
            })

            #Cập nhật trạng thái
            state = next_state
            current_data = next_data
            step_count += 1

            #Dừng nếu kết thúc episode hoặc quá 20 bước
            if done or step_count > 20:
                break
        
        #Thêm toàn bộ trải nghiệm vào bộ nhớ agent
        for exp in episode_experiences:
            agent.store_experience(**exp)

        #Cập nhật network nếu có đủ mẫu
        if len(agent.memory) >= 32:
            agent.update()
        
        #Lưu model tốt nhất nếu reward cao hơn trước:
        if total_reward > best_reward:
            best_reward = total_reward
            #print(best_reward)
            agent.save_model("ppo_handover_model.pth")

    return agent

def main():
    pingpong_threshold = 2.0 #second
    log_file_path = "File copy.log"
    action_spaces = {'A3_OFFSET': [1, 2, 4, 6, 8, 10, 12], 'TTT': [40, 80, 160, 320, 640, 1280]}
    
    # Tham số PPO
    ppo_config = {
        'lr': 3e-4,   #learning-rate
        'gamma': 1.2,  # hệ số chiết khấu reward
        'eps_clip': 0.2,  #độ giới hạn trong clipping ratio PPO (tránh đi lệch gradient xa)
        'k_epochs': 4,   # số lần cập nhật mỗi khi update.
        'c1': 0.5,    #trọng số cho loss của value function.
        'c2': 0.01    #trọng số cho entropy bonus (khuyến khích đa dạng hành động).
    }
    
    print("Starting PPO training...")

    try:
        df = parse_log_file(log_file_path)
        df = detect_ping_pong(df, pingpong_threshold)
        baseline_metrics = evaluate_performance(df)
        print("Baseline HO Fail Rate:", baseline_metrics['handover_failure_rate'])
    except Exception as e:
        print("Error reading log file:", e)
        return

    episodes=50       #Số lần agent lặp lại toàn bộ quá trình học từ đầu đến cuối qua môi trường
    model_path = "ppo_handover_models.pth"    #Đường dẫn model có sẵn

    if os.path.exists(model_path):
        print("Loading pretrained model...")
        agent = PPOAgent(state_dim=15, action_spaces=action_spaces, **ppo_config)
        agent.load_model(model_path)
    else:
        print("No pretrained model found. Training new agent...")
        agent = train_ppo_agent(df, action_spaces, ppo_config, episodes)
        print("Training completed.")    

    # === Xuất ra cặp tham số hiệu quả nhất ===
    print("\n=== Recommended Optimal Parameters ===")
    best_action_idx, _ = agent.select_action(agent.get_state_vector(df), explore=False)
    best_params = agent.action_index_to_params(best_action_idx)
    print(f"Recommended A3_OFFSET: {best_params['A3_OFFSET']} dB")
    print(f"Recommended TTT: {best_params['TTT']} ms")


if __name__ == "__main__":
    main()
