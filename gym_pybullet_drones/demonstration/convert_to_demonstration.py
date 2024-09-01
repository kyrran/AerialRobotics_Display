import pandas as pd
import numpy as np
import json
import os
from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
from scipy.interpolate import CubicSpline

bulletDroneEnv = BulletDroneEnv()

EXPR_BRANCH_POSITION = [-705.489013671875, 51.9111213684082, 1576.4576416015625]
interval_seconds = 0.05
NUM_DRONES = 1
from gym_pybullet_drones.utils.utils import str2bool, sync
import time


def calc_reward(state):
    x, y, z, t = state
    return bulletDroneEnv.calc_reward_and_done(np.array([x, y, z]), num_wraps=t)

def transform_demo(version, csv_file):
    # Load the CSV file
    df = pd.read_csv("./processed/" + csv_file)

    df['delta_drone_x'] = df['drone_x'].diff().fillna(0)
    df['delta_drone_y'] = df['drone_y'].diff().fillna(0)
    df['delta_drone_z'] = df['drone_z'].diff().fillna(0)
    
    df['distance'] = np.sqrt(df['delta_drone_x']**2 + df['delta_drone_y']**2 + df['delta_drone_z']**2)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ns')

    waypoints = []
    previous_time = df['Timestamp'].iloc[0]

    start_adding_waypoints = False
    initial_movement_found = False
    has_hit = False
    for index, row in df.iterrows():
        if row['drone_x'] - row['round_bar_x'] < 0:
            has_hit = True
        num_wraps = 1.0 if has_hit else 0.0
        
        if not start_adding_waypoints and row['drone_z'] > 2000:
            start_adding_waypoints = True
            prev_x = row['drone_x']
        
        if start_adding_waypoints and not initial_movement_found:
            delta_x = row['drone_x'] - prev_x
            if abs(delta_x) > 0.3 * 1000:
                print(f"Movement found {delta_x}")
                initial_movement_found = True
        
        if start_adding_waypoints and initial_movement_found:
            if (row['Timestamp'] - previous_time).total_seconds() >= interval_seconds:
                waypoints.append((row['drone_x'] - EXPR_BRANCH_POSITION[0]+0,
                                  row['drone_y'] - EXPR_BRANCH_POSITION[1]+0,
                                  row['drone_z'] - EXPR_BRANCH_POSITION[2] + 2700,
                                  num_wraps))
                previous_time = row['Timestamp']

    x_original, _, _, _ = waypoints[0]
    mult = 1 if x_original >= 0 else -1
    print(f"Angle: {csv_file} is {mult}")
    print(len(waypoints))
    # Convert waypoints to numpy arrays for easy manipulation
    waypoints = np.array(waypoints)
   
    # # Apply cubic spline interpolation to x, y, z coordinates
    # cs_x = CubicSpline(times, waypoints[:, 0])
    # cs_y = CubicSpline(times, waypoints[:, 1])
    # cs_z = CubicSpline(times, waypoints[:, 2])
    
    # # Generate smoother waypoints by sampling the cubic splines
    # finer_times = np.linspace(0, 1, 100)  # Increase the number of points
    # smooth_waypoints = np.vstack((cs_x(finer_times), cs_y(finer_times), cs_z(finer_times))).T

    
    # # Handle w separately by repeating the nearest value
    # w_values = np.interp(finer_times, times, waypoints[:, 3])  # Interpolate w as nearest value

    state_action_reward = []
    
    curr_x, curr_y, curr_z = waypoints[0][:3] / 1000
    curr_w = waypoints[0][3]
    max_action_magnitude = 0
    
    for i in range(len(waypoints) - 1):
        next_x, next_y, next_z, next_w = waypoints[i]
        next_x = next_x / 1000
        next_y = next_y / 1000
        next_z = next_z / 1000
        action_x = next_x - curr_x
        action_y = next_y - curr_y
        action_z = next_z - curr_z

        action_magnitude = np.sqrt(action_x**2 + action_y**2 + action_z**2)

        if action_magnitude > 0.25:
            print(f"Warning: Action magnitude exceeds 0.25 at index {i}. Magnitude: {action_magnitude}")

        if action_magnitude > max_action_magnitude:
            max_action_magnitude = action_magnitude

        action = np.array([next_x, next_y, next_z]).reshape((bulletDroneEnv.num_drones, -1))
        
        cur = np.array([curr_x, curr_y, curr_z]).reshape((bulletDroneEnv.num_drones, -1))
        # obs, _, _, _, _ = bulletDroneEnv.step(action)
        # bulletDroneEnv.render()

        # actual_x, actual_y, actual_z = bulletDroneEnv.get_drone_currrent_pos()
        waypoint = bulletDroneEnv._calculateNextStep(current_position=cur,destination=action)
        # print(f"hi:P{waypoint}")
        
        reward, done = calc_reward((curr_x, curr_y, curr_z, next_w))
       
        # action = (waypoint[0][0] - curr_x, waypoint[0][1] -curr_y, waypoint[0][2] - curr_z)

        state_action_reward.append(((curr_x, curr_y, curr_z, curr_w), (next_x, next_y, next_z), reward, (waypoint[0][0],waypoint[0][1],waypoint[0][2], next_w)))
        # state_action_reward.append(((curr_x, curr_y, curr_z, curr_w), (waypoint[0][0] - curr_x, waypoint[0][1] - curr_y, waypoint[0][2] - curr_z,), reward, (waypoint[0][0],waypoint[0][1],waypoint[0][2], next_w)))
        # state_action_reward.append(((curr_x, curr_y, curr_z, curr_w), (action_x, action_y, action_z), reward, (next_x,next_y,next_z, next_w)))
        
        
        # state_action_reward.append(((curr_x, curr_y, curr_z, curr_w), (action_x, action_y, action_z), reward, (next_x,next_y,next_z, next_w)))


        curr_x = waypoint[0][0]
        curr_y = waypoint[0][1]
        curr_z = waypoint[0][2]
        curr_w = next_w

        if done:
            break
        

    print(f"Largest action magnitude: {max_action_magnitude}")
    print("NUM_WAYPOINTS: " + str(len(state_action_reward)))

    state_action_reward_serializable = []

    for state, action, reward, next_state in state_action_reward:
        state_action_reward_serializable.append({
            "state": list(state),
            "action": list(action),
            "reward": reward,
            "next_state": list(next_state)
        })

    # Ensure the directory exists
    os.makedirs("rl_demos", exist_ok=True)

    # Save to JSON
    file_path = f"rl_demos/rl_demo_{version}.json"
    with open(file_path, 'w') as file:
        
        json.dump(state_action_reward_serializable, file, indent=4)

    print(f"Data saved to {file_path}")

    return file_path

# def play_demo(env, json_file):
#     """Load and play the demonstration from the JSON file."""
#     with open(json_file, 'r') as file:
#         demo_data = json.load(file)
#      # Initialize the drone's position to the first waypoint
#     bulletDroneEnv.reset(position=np.array(demo_data[0]['state'][:3]))

#     counter = 0
#     start_time = time.time()
    
   
#     for entry in demo_data:
#         counter += 1
#         # state = entry['state'] if counter == 1 else entry['next_state']
#         # action = np.array(state[:3])  # Extract x, y, z
#         action = np.array(entry['action'])
#         action = np.reshape(action, (env.num_drones, -1))
        
#         # Step the environment with the action
#         obs, reward, done, truncated, info = env.step(action)
#         print(f"action given is {action}")
#         # print(f"obs: {obs}")
        
#         # Render the environment
#         env.render()
        
#  # Synchronize the steps to avoid running too fast
#         sync(counter, start_time, env.CTRL_TIMESTEP)
        
        
#         # Check if the episode is done
#         if done or truncated:
#             print(f"at index: {counter}")
#             break

       
    
#     print(f"Demonstration steps executed: {counter}")

def play_demo(env, json_file):
    """Load and play the demonstration from the JSON file."""
    with open(json_file, 'r') as file:
        demo_data = json.load(file)

    # Initialize the drone's position to the first waypoint
    env.reset(position=np.array(demo_data[0]['state'][:3]))

    counter = 0
    start_time = time.time()
    
    for entry in demo_data:
        counter += 1
        # Extract the action (target position) from the demo data
        action = np.array(entry['action'])
        action = np.reshape(action, (env.num_drones, -1))
        # print(f"before step, where i am given:{env.get_drone_currrent_pos()}")
        # print(f"action given:{action}")
        # Step the environment with the action
        action = action - env.get_drone_currrent_pos()
        # print(f"action diff is {action}")
        obs, reward, done, truncated, info = env.step(action)
        # print(f"obs: {obs}")
        # print(f"afterS step, where i am given:{env.get_drone_currrent_pos()}")
        # Render the environment
        env.render()
        
        # Synchronize the steps to avoid running too fast
        sync(counter, start_time, env.CTRL_TIMESTEP)
        
        # Check if the episode is done
        if done or truncated:
            print(f"Terminated at index: {counter}")
            break

    print(f"Demonstration steps executed: {counter}")


# Example usage
csv_file = ["rosbag2_2024_05_22-17_00_56.csv", "rosbag2_2024_05_22-17_03_00.csv",
            "rosbag2_2024_05_22-17_20_43.csv", "rosbag2_2024_05_22-17_26_15.csv", 
            "rosbag2_2024_05_22-18_10_51.csv", "rosbag2_2024_05_22-18_16_45.csv"]



# csv_file = ["rosbag2_2024_05_22-17_00_56_filtered_normalized.csv", "rosbag2_2024_05_22-17_03_00_filtered_normalized.csv",
#             "rosbag2_2024_05_22-17_20_43_filtered_normalized.csv", "rosbag2_2024_05_22-17_26_15_filtered_normalized.csv", 
#             "rosbag2_2024_05_22-18_10_51_filtered_normalized.csv", "rosbag2_2024_05_22-18_16_45_filtered_normalized.csv"]


for i in range(len(csv_file)):
    print("--------------------------")
    json_file_path = transform_demo(i+1, csv_file[i])
    play_demo(bulletDroneEnv, json_file_path)
    
    print("--------------------------")
    
bulletDroneEnv.close()
