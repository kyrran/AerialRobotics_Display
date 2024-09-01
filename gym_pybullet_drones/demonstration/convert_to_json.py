import os
import json
import time
import numpy as np
import pandas as pd
from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
from gym_pybullet_drones.utils.utils import sync


# Constants
# EXPR_BRANCH_POSITION_DEMO = np.array([-0.705489013671875, 0.0519111213684082, 1.5764576416015625])


def transform_demo(env, version, file_path):
    """Transform the demonstration into a JSON file."""
    df = pd.read_csv(file_path)
    waypoints = []

    has_hit = False
    for _, row in df.iterrows():
        if row['drone_x'] < 0.0:
            has_hit = True
        num_wraps = 1.0 if has_hit else 0.0
        waypoints.append((row['drone_x'], row['drone_y'], row['drone_z'], num_wraps))

    # Add hover waypoints
    hover_position = [waypoints[0][0], waypoints[0][1], 0.0]
    # waypoints = add_hover_waypoints(hover_position, waypoints)

    return step_pair_calculation(env, waypoints, version)

def step_pair_calculation(env, waypoints, version):
    """Calculate state-action-reward pairs from waypoints with synchronization."""
    state_action_reward = []
    curr_x, curr_y, curr_z, curr_w = waypoints[0]

    env.reset(position=np.array([curr_x, curr_y, curr_z]))

    start_time = time.time()

    for i in range(1, len(waypoints)):
        next_x, next_y, next_z, next_w = waypoints[i]
        
        # if next_w == 1.0:
        #     print("yes")
        
        action = np.array([next_x, next_y, next_z]).reshape((env.num_drones, -1))
        
        # action = action - env.get_drone_currrent_pos()
 
        action_to_save = action.copy()
        # if env.get_drone_currrent_pos()[0] <= 0:
        #     env.step_size = .25
        
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        # print(f"reward after step():{reward}")
   
        reward, done = env.calc_reward_and_done((curr_x, curr_y, curr_z), next_w)
        
        action_diff_actual = (obs[0] -curr_x , obs[1]-curr_y, obs[2]-curr_z)
        
        # print(f"reward after calcu:{reward}")
        state_action_reward.append(((curr_x, curr_y, curr_z, curr_w), 
                                    (action_to_save[0][0], action_to_save[0][1], action_to_save[0][2]), reward, 
                                    (obs[0], obs[1], obs[2], next_w)))
        
        
        # state_action_reward.append(((curr_x, curr_y, curr_z, curr_w), 
        #                             (action_diff_actual[0],action_diff_actual[1],action_diff_actual[2]), reward, 
        #                             (obs[0], obs[1], obs[2], next_w)))
        
        
        curr_x, curr_y, curr_z, curr_w = obs[0], obs[1], obs[2], next_w

        if done:
            print("Simulation completed.")
            break
        
        if truncated:
            print("early stopped!")
            break

        sync(i, start_time, env.CTRL_TIMESTEP)

    return save_to_json(state_action_reward, version)

def save_to_json(state_action_reward, version):
    """Save state-action-reward pairs to a JSON file."""
    # Convert NumPy types to Python types
    state_action_reward_serializable = [{
        "state": [float(x) for x in state],
        "action": [float(a) for a in action],
        "reward": float(reward),
        "next_state": [float(x) for x in next_state]
    } for state, action, reward, next_state in state_action_reward]

    os.makedirs("rl_demos_new", exist_ok=True)
    file_path = f"rl_demos_new/rl_demo_new_{version}.json"

    with open(file_path, 'w') as file:
        json.dump(state_action_reward_serializable, file, indent=4)

    print(f"Data saved to {file_path}")
    return file_path

def main():
    """Main function to process and transform demonstrations."""
    bulletDroneEnv = BulletDroneEnv(client=False)

    # csv_files = [
    #     "rosbag2_2024_05_22-17_00_56.csv", 
    #     "rosbag2_2024_05_22-17_03_00.csv",
    #     "rosbag2_2024_05_22-17_20_43.csv", 
    #     "rosbag2_2024_05_22-17_26_15.csv", 
    #     "rosbag2_2024_05_22-18_10_51.csv", 
    #     "rosbag2_2024_05_22-18_16_45.csv"
    # ]
    
    csv_files = ["rosbag2_2024_05_22-17_00_56_filtered_normalized.csv", "rosbag2_2024_05_22-17_03_00_filtered_normalized.csv",
            "rosbag2_2024_05_22-17_20_43_filtered_normalized.csv", "rosbag2_2024_05_22-17_26_15_filtered_normalized.csv", 
            "rosbag2_2024_05_22-18_10_51_filtered_normalized.csv", "rosbag2_2024_05_22-18_16_45_filtered_normalized.csv"]


    for i, csv_file in enumerate(csv_files):
        print(f"------ Processing {csv_file} ------")

        file_path = f"./optimised/reduced_{csv_file}"
        json_file_path = transform_demo(bulletDroneEnv, i + 1, file_path)

        print("-------- Done ------------------")

if __name__ == "__main__":
    main()
