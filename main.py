#pip install matplotlib
from parse import parse_log_file
from detect_pingpong import detect_ping_pong
from ppo_agent import PPOAgent, HandoverEnvironment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

def evaluate_performance(df: pd.DataFrame) -> dict:
    """Đánh giá hiệu suất của tham số handover"""
    if len(df) == 0:
        return {
            'handover_failure_rate': 0,
            'ping_pong_rate': 0,
            'avg_rsrp_source': 0,
            'avg_rsrp_target': 0,
            'total_handovers': 0
        }
    
    metrics = {
        'handover_failure_rate': df['HOF'].mean(),
        'ping_pong_rate': df['PingPong'].mean(),
        'avg_rsrp_source': df['RSRP_SOURCE'].mean(),
        'avg_rsrp_target': df['RSRP_TARGET'].mean(),
        'total_handovers': len(df),
        'avg_ho_margin': df['HO_MARGIN'].mean(),
        'avg_ttt': df['TIME_TO_TRIGGER'].mean()
    }
    
    return metrics

def plot_training_results(training_log: list, save_path: str = None):
    """Vẽ biểu đồ kết quả training"""
    if not training_log:
        return
    
    episodes = [log['episode'] for log in training_log]
    rewards = [log['total_reward'] for log in training_log]
    ho_failure_rates = [log['metrics']['handover_failure_rate'] for log in training_log]
    ping_pong_rates = [log['metrics']['ping_pong_rate'] for log in training_log]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward over episodes
    ax1.plot(episodes, rewards)
    ax1.set_title('Total Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Handover failure rate
    ax2.plot(episodes, ho_failure_rates, color='red')
    ax2.set_title('Handover Failure Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Failure Rate')
    ax2.grid(True)
    
    # Ping-pong rate
    ax3.plot(episodes, ping_pong_rates, color='orange')
    ax3.set_title('Ping-Pong Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Ping-Pong Rate')
    ax3.grid(True)
    
    # A3 Offset and TTT evolution
    a3_offsets = [log['action_params']['A3_OFFSET'] for log in training_log]
    ttts = [log['action_params']['TTT'] for log in training_log]
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(episodes, a3_offsets, 'b-', label='A3 Offset')
    line2 = ax4_twin.plot(episodes, ttts, 'g-', label='TTT')
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('A3 Offset (dB)', color='b')
    ax4_twin.set_ylabel('TTT (ms)', color='g')
    ax4.set_title('Parameter Evolution')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training results plot saved to {save_path}")
    
    plt.show()

def train_ppo_agent(df: pd.DataFrame, action_spaces: dict, 
                   episodes: int = 100, save_model_path: str = None):
    """Training PPO agent"""
    
    # Khởi tạo environment và agent
    env = HandoverEnvironment(df, window_size=50)
    state_dim = 15  # Số features trong state vector
    agent = PPOAgent(state_dim=state_dim, action_spaces=action_spaces)
    
    training_log = []
    best_reward = float('-inf')
    
    print("Start training PPO agent...")
    print(f"Action spaces: {action_spaces}")
    print(f"Total action combinations: {agent.action_dim}")
    
    for episode in range(episodes):
        # Reset environment
        current_data = env.reset()
        state = agent.get_state_vector(current_data)
        
        total_reward = 0
        step_count = 0
        episode_experiences = []
        
        while True:
            # Chọn action
            action_idx, log_prob = agent.select_action(state, explore=True)
            action_params = agent.action_index_to_params(action_idx)
            
            # Thực hiện action trong environment
            prev_data = current_data.copy()
            next_data, done = env.step(action_params)
            next_state = agent.get_state_vector(next_data)
            
            # Tính reward
            reward = agent.calculate_reward(prev_data, next_data, action_params)
            total_reward += reward
            
            # Lưu experience
            episode_experiences.append({
                'state': state,
                'action': action_idx,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob
            })
            
            state = next_state
            current_data = next_data
            step_count += 1
            
            if done or step_count > 20:  # Giới hạn số bước
                break
        
        # Thêm experiences vào memory
        for exp in episode_experiences:
            agent.store_experience(**exp)
        
        # Update agent
        if len(agent.memory) >= 32:
            agent.update()
        
        # Đánh giá performance
        final_metrics = evaluate_performance(current_data)
        
        # Log kết quả
        episode_log = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'steps': step_count,
            'action_params': action_params,
            'metrics': final_metrics
        }
        training_log.append(episode_log)
        
        # Save best model
        if total_reward > best_reward and save_model_path:
            best_reward = total_reward
            agent.save_model(save_model_path)
            print(f"New best model saved with reward: {total_reward:.2f}")
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  HO Failure Rate: {final_metrics['handover_failure_rate']:.3f}")
            print(f"  Ping-Pong Rate: {final_metrics['ping_pong_rate']:.3f}")
            print(f"  Action: A3_OFFSET={action_params['A3_OFFSET']}, TTT={action_params['TTT']}")
            print("-" * 50)
    
    return agent, training_log

def test_agent(df: pd.DataFrame, agent: PPOAgent, action_spaces: dict, test_episodes: int = 10):
    """Test trained agent"""
    env = HandoverEnvironment(df, window_size=50)
    test_results = []
    
    print("Testing trained agent...")
    
    for episode in range(test_episodes):
        current_data = env.reset()
        state = agent.get_state_vector(current_data)
        
        total_reward = 0
        actions_taken = []
        
        while True:
            # Chọn action (không explore)
            action_idx, _ = agent.select_action(state, explore=False)
            action_params = agent.action_index_to_params(action_idx)
            actions_taken.append(action_params)
            
            # Thực hiện action
            prev_data = current_data.copy()
            next_data, done = env.step(action_params)
            next_state = agent.get_state_vector(next_data)
            
            # Tính reward
            reward = agent.calculate_reward(prev_data, next_data, action_params)
            total_reward += reward
            
            state = next_state
            current_data = next_data
            
            if done:
                break
        
        # Đánh giá performance
        metrics = evaluate_performance(current_data)
        
        test_result = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'final_metrics': metrics,
            'actions_taken': actions_taken
        }
        test_results.append(test_result)
        
        print(f"Test Episode {episode + 1}: Reward={total_reward:.2f}, "
              f"HO_Fail={metrics['handover_failure_rate']:.3f}, "
              f"PingPong={metrics['ping_pong_rate']:.3f}")
    
    return test_results

def main():
 # Cấu hình
    pingpong_threshold = 2.0  # giây
    log_file_path = "File.log"
    
    # Action spaces cho A3 offset và TTT
    action_spaces = {
        'A3_OFFSET': [0, 2, 4, 6, 8, 10, 12],  # dB
        'TTT': [40, 80, 160, 320, 640, 1280]   # ms
    }

    print("=== 5G Handover Optimization with PPO ===")
    print(f"Action spaces: {action_spaces}")

    # Đọc và xử lý dữ liệu
    print("\n1. Reading and processing log data...")
    try:
        df = parse_log_file(log_file_path)
        df = detect_ping_pong(df, pingpong_threshold)
        print(f"Loaded {len(df)} handover records")
        
        # Đánh giá baseline performance
        baseline_metrics = evaluate_performance(df)
        print("\nBaseline Performance:")
        for metric, value in baseline_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    except Exception as e:
        print(f"Bugs: {e}")

    # 2. Training PPO Agent
    print("\n2. Training PPO Agent...")
    training_episodes = 100
    model_save_path = "ppo_handover_model.pth"
    
    agent, training_log = train_ppo_agent(
        df=df,
        action_spaces=action_spaces,
        episodes=training_episodes,
        save_model_path=model_save_path
    )
    
    # 3. Vẽ kết quả training
    print("\n3. Plotting training results...")
    plot_training_results(training_log, save_path="training_results.png")
    
    # 4. Test agent đã train
    print("\n4. Testing trained agent...")
    test_results = test_agent(df, agent, action_spaces, test_episodes=10)
    
    # 5. So sánh kết quả
    print("\n5. Performance Comparison:")
    print("=" * 60)
    
    # Baseline metrics
    print("BASELINE PERFORMANCE:")
    for metric, value in baseline_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # PPO agent average performance
    avg_test_metrics = {}
    for metric in baseline_metrics.keys():
        avg_test_metrics[metric] = np.mean([result['final_metrics'][metric] for result in test_results])
    
    print("\nPPO AGENT AVERAGE PERFORMANCE:")
    for metric, value in avg_test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nIMPROVEMENT:")
    for metric in baseline_metrics.keys():
        if baseline_metrics[metric] != 0:
            improvement = ((baseline_metrics[metric] - avg_test_metrics[metric]) / baseline_metrics[metric]) * 100
            print(f"  {metric}: {improvement:+.2f}%")
    
    # 6. Lưu kết quả
    print("\n6. Saving results...")
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'action_spaces': action_spaces,
        'training_episodes': training_episodes,
        'baseline_metrics': baseline_metrics,
        'ppo_average_metrics': avg_test_metrics,
        'training_log': training_log,
        'test_results': test_results
    }
    
    with open('handover_optimization_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # 7. Recommend optimal parameters
    print("\n7. Recommended Optimal Parameters:")
    print("=" * 60)
    
    # Tìm best action từ test results
    best_test = max(test_results, key=lambda x: x['total_reward'])
    best_actions = best_test['actions_taken']
    
    if best_actions:
        # Lấy action cuối cùng (thường là tốt nhất)
        recommended_params = best_actions[-1]
        print(f"Recommended A3_OFFSET: {recommended_params['A3_OFFSET']} dB")
        print(f"Recommended TTT: {recommended_params['TTT']} ms")
        
        print(f"\nWith these parameters:")
        final_metrics = best_test['final_metrics']
        print(f"  Expected HO Failure Rate: {final_metrics['handover_failure_rate']:.3f}")
        print(f"  Expected Ping-Pong Rate: {final_metrics['ping_pong_rate']:.3f}")
        print(f"  Expected Total Reward: {best_test['total_reward']:.2f}")
    
    # 8. Near-RT RIC Integration Guide
    print("\n8. Near-RT RIC Integration Guide:")
    print("=" * 60)
    print("To integrate with Near-RT RIC:")
    print("1. Implement xApp service using O-RAN interfaces")
    print("2. Subscribe to E2 messages for real-time RAN data")
    print("3. Use trained PPO agent to determine optimal parameters")
    print("4. Send parameter updates via E2 interface to gNB")
    print("5. Monitor performance and retrain agent periodically")
    
    print(f"\nModel saved to: {model_save_path}")
    print("Training completed successfully!")

def simulate_near_rt_ric_integration(agent: PPOAgent, action_spaces: dict):
    """
    Mô phỏng tích hợp với Near-RT RIC
    Trong thực tế, đây sẽ là xApp giao tiếp qua E2 interface
    """
    print("\n=== Near-RT RIC Integration Simulation ===")
    
    # Simulate receiving real-time data from RAN
    print("1. Receiving real-time handover data from E2 interface...")
    
    # Simulate current network state
    current_state = np.random.rand(15)  # Mock state vector
    
    # Agent recommends action
    print("2. PPO Agent analyzing network state...")
    action_idx, confidence = agent.select_action(current_state, explore=False)
    recommended_params = agent.action_index_to_params(action_idx)
    
    print("3. Recommended parameter updates:")
    print(f"   A3_OFFSET: {recommended_params['A3_OFFSET']} dB")
    print(f"   TTT: {recommended_params['TTT']} ms")
    print(f"   Confidence: {np.exp(confidence):.3f}")
    
    # Simulate sending updates to gNB
    print("4. Sending parameter updates to gNB via E2 interface...")
    print("   - RIC Control Request sent")
    print("   - Parameters updated successfully")
    
    # Simulate monitoring
    print("5. Monitoring handover performance...")
    print("   - Collecting new handover statistics")
    print("   - Performance improvement detected")
    
    return recommended_params

main()

