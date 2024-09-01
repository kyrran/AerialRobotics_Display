import argparse
import numpy as np
import time
import os
import pandas as pd
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync
from process_trajectory import toppra_waypoints
from gym_pybullet_drones.envs.TetherModelSimulationEnvPID import TetherModelSimulationEnvPID
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from rdp import rdp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# Set the Seaborn style
sns.set(style="whitegrid", palette="pastel", color_codes=True)

def plot_trajectories(original, reduced, smoothed):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original trajectory
    # ax.plot(original[:, 0], original[:, 1], original[:, 2], label='Original Trajectory', color=sns.color_palette()[0], linewidth=2)

    # Plot the reduced trajectory
    ax.plot(reduced[:, 0], reduced[:, 1], reduced[:, 2], label='Reduced Trajectory', color=sns.color_palette()[1], linewidth=2)

    # Plot the smoothed trajectory
    ax.plot(smoothed[:, 0], smoothed[:, 1], smoothed[:, 2], label='Smoothed Trajectory', color=sns.color_palette()[2], linewidth=2)

    # Determine the combined axis limits
    # all_data = np.concatenate((original, reduced, smoothed), axis=0)
    # range_min = np.min(all_data)
    # range_max = np.max(all_data)

    # # Set the same limits for all axes
    # ax.set_xlim(range_min, range_max)
    # ax.set_ylim(range_min, range_max)
    # ax.set_zlim(range_min, range_max)

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Comparison of Original, Reduced, and Smoothed Trajectories')

    ax.legend()
    plt.show()


def reduce_waypoints(waypoints, epsilon=0.05):
    simplified_waypoints = np.array(rdp(waypoints.tolist(), epsilon=epsilon))
    smoothed_waypoints_2 = toppra_waypoints(simplified_waypoints, min(len(simplified_waypoints) * 20, 200))
    smoothed_waypoints_2 = smoothed_waypoints_2[['drone_x', 'drone_y', 'drone_z']].values
    # Plot the trajectories
    plot_trajectories(waypoints, simplified_waypoints, smoothed_waypoints_2)
        
    return smoothed_waypoints_2

def add_hover_waypoints(hover_position, trajectory_points, control_freq_hz=48, period=10, hover_height=2.0):
    num_wp = control_freq_hz * period
    ascent_duration = int(num_wp / 4)
    target_pos = np.zeros((num_wp + len(trajectory_points), 3))
    
    # Calculate the difference between the current z and the hover z
    initial_z = hover_position[2]
    final_z = max(initial_z, hover_height)  # Hover at the initial position if it's higher than hover height

    for i in range(num_wp):
        if i < ascent_duration and initial_z < hover_height:
            # Gradually move the drone to the hover altitude only if it's below the hover height
            target_pos[i, :] = [hover_position[0], hover_position[1], initial_z + (final_z - initial_z) * (i / ascent_duration)]
        else:
            # Keep it at the hover altitude or the initial altitude if it's already higher
            target_pos[i, :] = [hover_position[0], hover_position[1], final_z]
    
    # Append the actual trajectory points after the hover phase
    target_pos[num_wp:num_wp + len(trajectory_points), :] = trajectory_points
    return target_pos

def run_simulation(simulation, target_pos, start):
    init = start.copy()
    action = np.zeros((simulation.num_drones, 3))
    action = init.reshape(action.shape)
    start_time = time.time()
    
    for i in range(0, int(simulation.duration_sec * simulation.control_freq_hz)):
       
        obs, reward, terminated, truncated, info = simulation.step(action)
       
        for j in range(simulation.num_drones):
            wp_index = min(simulation.wp_counters[j], len(target_pos) - 1)
            
            action[j, :] = target_pos[wp_index, :]
            
            
        simulation.wp_counters += 1
        if np.all(simulation.wp_counters > len(target_pos)):
            break
        
        act2 = action[j, :].squeeze()
        for j in range(simulation.num_drones):
            simulation.logger.log(drone=j,
                                  timestamp=i / simulation.control_freq_hz,
                                  state=np.hstack([obs[j][0:3],
                                                   np.zeros(4),
                                                   obs[j][3:15],
                                                   act2[j]
                                                   ]),
                                  control=np.zeros(12),
                                  payload_position=simulation.weight.get_position()
                                  )
            
        simulation.render()
        if simulation.gui:
            sync(i, start_time, simulation.CTRL_TIMESTEP)
    
    # simulation.logger.save()
    # simulation.logger.save_as_csv("pid")
    # if simulation.plot:
    #     simulation.logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone simulation using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone', default=DroneModel("cf2x"), type=DroneModel, help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=1, type=int, help='Number of drones (default: 1)', metavar='')
    parser.add_argument('--physics', default=Physics("pyb"), type=Physics, help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=True, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=False, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot', default=True, type=str2bool, help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=False, type=str2bool, help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles', default=False, type=str2bool, help='Whether to add obstacles to the environment (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=48, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=60, type=int, help='Duration of the simulation in seconds (default: 60)', metavar='')
    parser.add_argument('--output_folder', default='results', type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=False, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--hover_height', default=2.0, type=float, help='Hover height in meters (default: 2.0)', metavar='')
    parser.add_argument('--csv_files', nargs='+', default=['trajectory_1.csv'], type=str, help='List of CSV files with trajectory points', metavar='')

    args = parser.parse_args()

    # Initialize the simulation environment once
    simulation = TetherModelSimulationEnvPID(
        start_pos=np.array([0, 0, 0]),  # Initial placeholder start position
        drone=args.drone,
        num_drones=args.num_drones,
        physics=args.physics,
        gui=args.gui,
        record_video=args.record_video,
        plot=args.plot,
        user_debug_gui=args.user_debug_gui,
        obstacles=args.obstacles,
        simulation_freq_hz=args.simulation_freq_hz,
        control_freq_hz=args.control_freq_hz,
        duration_sec=args.duration_sec,
        output_folder=args.output_folder,
        colab=args.colab,
        hover_height=args.hover_height,
        obs=ObservationType.KIN, 
        act=ActionType.PID,
    )
    
    
    
    csv_files = ["rosbag2_2024_05_22-17_00_56.csv", "rosbag2_2024_05_22-17_03_00.csv",
            "rosbag2_2024_05_22-17_20_43.csv", "rosbag2_2024_05_22-17_26_15.csv", 
            "rosbag2_2024_05_22-18_10_51.csv", "rosbag2_2024_05_22-18_16_45.csv"]

    
    # Loop over all provided CSV files
    for csv_file in csv_files:
        data = pd.read_csv("../demonstration/processed/"+csv_file)
        
        start = np.array([data['drone_x'][0] / 1000, data['drone_y'][0] / 1000, 0])
        data['drone_x'] = data['drone_x'] / 1000.00 - (-0.705489013671875)
        data['drone_y'] = data['drone_y'] / 1000.00 - 0.0519111213684082
        data['drone_z'] = data['drone_z'] / 1000.00 - 1.5764576416015625 + 2.7
        data = data[data['drone_z'] > 2.3]
        
        
        
        waypoints = data[['drone_x', 'drone_y', 'drone_z']].values
        # start = waypoints[0]
        reduced_trajectory_points = reduce_waypoints(waypoints, epsilon=0.05)
        trajectory_points = add_hover_waypoints(hover_position=[start[0], start[1], 0], trajectory_points=reduced_trajectory_points)
        
        
        
            # Reset the environment with the new start position
        simulation.reset(pos=start)
        
        # Run the simulation with the new waypoints
        run_simulation(simulation, trajectory_points, start)
        print(f"---------{csv_file} has finished------------------------")
    # Close the simulation environment after all files are processed
    simulation.close()