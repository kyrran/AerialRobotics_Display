from gym_pybullet_drones.envs.wrappers.hovering_wrapper import HoveringWrapper

from gym_pybullet_drones.envs.wrappers.symmetric_wrapper import SymmetricWrapper

from gym_pybullet_drones.envs.wrappers.position_wrapper import PositionWrapper

from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv

from gym_pybullet_drones.algorithms.sacfd import SACfD
from gym_pybullet_drones.algorithms.dual_buffer import DualReplayBuffer
from stable_baselines3 import SAC,PPO
import numpy as np
import time
from gym_pybullet_drones.utils.utils import str2bool, sync

import csv
import os

def test_agent(agent, env, save_directory, num_episodes=5):
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = np.array(obs) 
        start_time = time.time()  # Start time of the episode
        done = False
        total_reward = 0
        counter = 0
        
        episode_positions = []  # To store positions and times for this episode
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        
            # Get drone and payload positions
            drone_position = obs[:3]  # Assuming the first 3 values in obs are the drone's x, y, z positions
            payload_position = env.get_wrapper_attr('weight').get_position()
            
            # Calculate the real-world time elapsed since the start of the episode
            real_time_elapsed = time.time() - start_time
            
            # Append both drone and payload positions along with real time to the episode list
            episode_positions.append((real_time_elapsed, drone_position, payload_position))
            
            total_reward += reward
            env.render()
            sync(counter, start_time, env.get_wrapper_attr('CTRL_TIMESTEP'))
            if done or truncated:
                break
            counter += 1
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        # Save positions and real time for this episode to a separate CSV file
        episode_save_path = os.path.join(save_directory, f"drone_payload_positions_episode_{episode + 1}.csv")
        with open(episode_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Real_Time_Elapsed', 'Drone_X', 'Drone_Y', 'Drone_Z', 'Payload_X', 'Payload_Y', 'Payload_Z'])
            for timestep, (real_time, drone_position, payload_position) in enumerate(episode_positions):
                writer.writerow([real_time] + list(drone_position) + list(payload_position))
    
    env.close()

# Example path for model and save directory
# path = "/home/kangle/Documents/FYP/gym-pybullet-drones/gym_pybullet_drones/examples/models/save-algorithm-SAC-09.11.2024_01.48.12-1200000/"

# path = "/home/kangle/Documents/FYP/gym-pybullet-drones/gym_pybullet_drones/examples/models/fix_errer_new_reward+3_distance/save-algorithm-SAC-09.14.2024_13.13.40-1200000_+3_fix_distance/"
# path = "/home/kangle/Documents/FYP/gym-pybullet-drones/gym_pybullet_drones/examples/models/expr/save-algorithm-SAC-09.14.2024_13.13.40-1200000_+3_fix_distance/"

path = "/home/kangle/Documents/FYP/gym-pybullet-drones/gym_pybullet_drones/examples/models/400-episode/save-algorithm-SAC-09.15.2024_17.06.05-120000/"
model_path = path + "best_model.zip"
save_directory = path + "episode_positions/"

# Load the model
model = SAC.load(model_path)

# Create the test environment
test_env = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(gui=True))))

# Test the agent and save positions for each episode in a separate file
test_agent(agent=model, env=test_env, save_directory=save_directory)
