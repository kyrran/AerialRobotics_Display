import numpy as np
import time
import pandas as pd
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ObservationType, ActionType
from gym_pybullet_drones.envs.TetherModelSimulationEnvPID import TetherModelSimulationEnvPID
import pybullet as p
from gym_pybullet_drones.utils.utils import str2bool, sync

# Settings for the simulation
DRONE_MODEL = DroneModel("cf2x")
NUM_DRONES = 1
PHYSICS = Physics("pyb")
GUI = True
RECORD_VIDEO = False
PLOT = True
USER_DEBUG_GUI = False
OBSTACLES = False
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 240
DURATION_SEC = 60
OUTPUT_FOLDER = 'results'
HOVER_HEIGHT = 2.0

# CSV trajectory file
TRAJECTORY_FILE = '/home/kangle/Documents/FYP/gym-pybullet-drones/gym_pybullet_drones/examples/models/expr/save-algorithm-SACfD-09.13.2024_12.20.51-1200000_6_demos_+3_new/reduced_waypoints/drone_payload_positions_episode_4_simplified_drone.csv'

def load_waypoints_from_csv(trajectory_file):
    """Load waypoints directly from the CSV file."""
    trajectory_data = pd.read_csv(trajectory_file)
    waypoints = trajectory_data[['Drone_X', 'Drone_Y', 'Drone_Z']].values
    return waypoints

def run_simulation(simulation, target_pos):
    """Run the drone simulation with the loaded waypoints."""
    
    action = np.zeros((simulation.num_drones, 3))
    start_time = time.time()
    
    for i in range(0, 1200):
       
        for j in range(simulation.num_drones):
            wp_index = min(simulation.wp_counters[j], len(target_pos) - 1)
            action[j, :] = target_pos[wp_index, :]
            obs, reward, terminated, truncated, info = simulation.step(action)
            print(f"Waypoint {wp_index}: Target - {action[j, :]}")
        
        simulation.wp_counters += 1
        
        # If all drones have finished the waypoint list, stop the simulation
        # if np.all(simulation.wp_counters > len(target_pos)):
        #     break
        
        simulation.render()
        if simulation.gui:
            sync(i, start_time, simulation.CTRL_TIMESTEP)
    
    # Save and close the simulation log
    simulation.close()
    # simulation.logger.save()
    # simulation.logger.save_as_csv("pid")
    # if simulation.plot:
    #     simulation.logger.plot()

if __name__ == "__main__":
    
    # Load waypoints directly from the CSV file
    target_pos = load_waypoints_from_csv(TRAJECTORY_FILE)
    
    # Set the initial start position as the first position from the waypoints
    START_POS = target_pos[0]  # Use the first waypoint as the initial position

    # Initialize the simulation environment with the new starting position
    simulation = TetherModelSimulationEnvPID(start_pos=START_POS,
                                             drone=DRONE_MODEL,
                                             num_drones=NUM_DRONES,
                                             physics=PHYSICS,
                                             gui=GUI,
                                             record_video=RECORD_VIDEO,
                                             plot=PLOT,
                                             user_debug_gui=USER_DEBUG_GUI,
                                             obstacles=OBSTACLES,
                                             simulation_freq_hz=SIMULATION_FREQ_HZ,
                                             control_freq_hz=20,
                                             duration_sec=DURATION_SEC,
                                             output_folder=OUTPUT_FOLDER,
                                             hover_height=HOVER_HEIGHT,
                                             obs=ObservationType.KIN,
                                             act=ActionType.PID)

    # Run the simulation with the loaded waypoints
    run_simulation(simulation, target_pos)
