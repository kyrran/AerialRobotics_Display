import pandas as pd
import numpy as np
import json
import os
from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
from rdp import rdp
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

EXPR_BRANCH_POSITION_demo = [-0.705489013671875, 0.0519111213684082, 1.5764576416015625]

NUM_DRONES = 1

def pre_process_waypoints(filename):
    """Load and preprocess waypoints from a CSV file."""
    data = pd.read_csv("./filtered_rosbags/" + filename)
    
    data['drone_x'] = data['drone_x'] / 1000.00 - EXPR_BRANCH_POSITION_demo[0]
    data['drone_y'] = data['drone_y'] / 1000.00 - EXPR_BRANCH_POSITION_demo[1]
    data['drone_z'] = data['drone_z'] / 1000.00 - EXPR_BRANCH_POSITION_demo[2] + 2.7
    
    # data = data[data['drone_z'] > 2.0]
    
    waypoints = data[['drone_x', 'drone_y', 'drone_z']].values
    
    return waypoints

def reduce_waypoints(waypoints, epsilon=0.05):
    """Simplify the trajectory using Ramer-Douglas-Peucker (RDP) algorithm."""
    simplified_waypoints = rdp(waypoints.tolist(), epsilon=epsilon)
    print(len( simplified_waypoints))
    return np.array(simplified_waypoints)

def smooth_waypoints(simplified_waypoints, filename):
    smoothed_waypoints = toppra_waypoints(simplified_waypoints, min(len(simplified_waypoints)*50,300),filename)
    smoothed_waypoints = smoothed_waypoints[['drone_x', 'drone_y', 'drone_z']].values
    print(f"waypoints no:{len(smoothed_waypoints)}")
    return simplified_waypoints

def toppra_waypoints(simplified_waypoints, num_samples, filename):
    
    """Toppra Algorithm to smooth the trajectory"""
    
    # Extract x, y, z columns for processing
    waypoints = simplified_waypoints
    
    # Define the velocity and acceleration limits
    vlim = np.array([1, 1, 1])  # velocity limits in each axis
    alim = np.array([0.5, 0.5, 0.5])  # acceleration limits in each axis

    # Create path from waypoints
    path = ta.SplineInterpolator(np.linspace(0, 1, len(waypoints)), waypoints)

    # Create velocity and acceleration constraints
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)

    # Setup the parameterization problem
    instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')

    # Compute the trajectory
    jnt_traj = instance.compute_trajectory(0, 0)

    # Sample the trajectory
    N_samples = num_samples
    ss = np.linspace(0, jnt_traj.duration, N_samples)
    qs = jnt_traj(ss)

    # Extract the x, y, z components of the trajectory
    x = qs[:, 0]
    y = qs[:, 1]
    z = qs[:, 2]

    # Save the processed waypoints to a new CSV file
    processed_waypoints = pd.DataFrame({
        'drone_x': x,
        'drone_y': y,
        'drone_z': z
    })
    
    os.makedirs("optimised", exist_ok=True)
    processed_file_path = "./optimised/reduced_" + filename
    processed_waypoints.to_csv(processed_file_path, index=False)
    
    return processed_waypoints


# def add_hover_waypoints(file_path, hover_position, trajectory_points, control_freq_hz=48, period=5):
#     """Add hover waypoints to the trajecthovenvory."""
#     num_wp = control_freq_hz * period
#     ascent_duration = int(num_wp / 4)
#     target_pos = np.zeros((num_wp + len(trajectory_points), 3))
    
#     for i in range(num_wp):
#         if i < 12:
#             target_pos[i, :] = [hover_position[0], hover_position[1], (hover_position[2] / ascent_duration) * i]
#         else:
#             target_pos[i, :] = hover_position
    
#     target_pos[num_wp:num_wp + len(trajectory_points), :] = trajectory_points
    
#     target_pos = trajectory_points
#     # Save the reduced waypoints to a new CSV file
#     reduced_waypoints = pd.DataFrame(target_pos, columns=['drone_x', 'drone_y', 'drone_z'])
#     os.makedirs("optimised", exist_ok=True)
#     reduced_waypoints_file = "./optimised/reduced_" + file_path
#     reduced_waypoints.to_csv(reduced_waypoints_file, index=False)
    
#     return target_pos, reduced_waypoints_file


# Example usage
# csv_file = ["rosbag2_2024_05_22-17_00_56.csv", "rosbag2_2024_05_22-17_03_00.csv",
#             "rosbag2_2024_05_22-17_20_43.csv", "rosbag2_2024_05_22-17_26_15.csv", 
#             "rosbag2_2024_05_22-18_10_51.csv", "rosbag2_2024_05_22-18_16_45.csv"]

csv_file = ["rosbag2_2024_05_22-17_00_56_filtered_normalized.csv", "rosbag2_2024_05_22-17_03_00_filtered_normalized.csv",
            "rosbag2_2024_05_22-17_20_43_filtered_normalized.csv", "rosbag2_2024_05_22-17_26_15_filtered_normalized.csv", 
            "rosbag2_2024_05_22-18_10_51_filtered_normalized.csv", "rosbag2_2024_05_22-18_16_45_filtered_normalized.csv"]


for i in range(len(csv_file)):
    print(f"------processing {csv_file[i]}------")
    waypoints = pre_process_waypoints(csv_file[i])
    reduced_trajectory_points = reduce_waypoints(waypoints, epsilon=0.01)
    smooth_waypoints(reduced_trajectory_points, csv_file[i])
    # target_pos, reduced_file_path = add_hover_waypoints(csv_file[i], hover_position=[reduced_trajectory_points[0][0], reduced_trajectory_points[0][1], max(2,reduced_trajectory_points[0][2])], trajectory_points=reduced_trajectory_points)
    print(f"---------Done----------")
