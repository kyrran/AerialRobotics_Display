import argparse
import numpy as np
import time
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync
from process_trajectory import load_waypoints, process_trajectory,process_trajectory_with_cubic_spline
from gym_pybullet_drones.envs.TetherModelSimulationEnvPID import TetherModelSimulationEnvPID
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

import pandas as pd
from rdp import rdp
import pybullet as p


def load_and_process_trajectory(trajectory_file, processed_trajectory_file, hover_height, control_freq_hz, start_pos):
    
    
    process_trajectory_with_cubic_spline(trajectory_file, processed_trajectory_file, [start_pos[0], start_pos[1], hover_height])
    trajectory_points = load_waypoints(processed_trajectory_file)
    
    period = 6
    num_wp = control_freq_hz * period
    ascent_duration = int(num_wp / 4)
    target_pos = np.zeros((num_wp + len(trajectory_points), 3))
    
    for i in range(num_wp):
        if i < ascent_duration:
            target_pos[i, :] = [start_pos[0], start_pos[1], (hover_height / ascent_duration) * i]
        else:
            target_pos[i, :] = [start_pos[0], start_pos[1], hover_height]
    
    target_pos[num_wp:num_wp + len(trajectory_points), :] = trajectory_points
    return target_pos

def run_simulation(simulation, target_pos):

    
    action = np.zeros((simulation.num_drones, 3))
    start_time = time.time()
    
    for i in range(0, int(simulation.duration_sec * simulation.control_freq_hz)):
       
        # obs, reward, terminated, truncated, info = simulation.step(action)
        for j in range(simulation.num_drones):
            wp_index = min(simulation.wp_counters[j], len(target_pos) - 1)
            action[j, :] = target_pos[wp_index, :]
            # action[j, :] = target_pos[wp_index, :] - simulation.get_drone_currrent_pos()
            obs, reward, terminated, truncated, info  = simulation.step(action)
            print(wp_index)
            # action[j, :], _, _ = simulation.ctrl[j].computeControlFromState(control_timestep=simulation.CTRL_TIMESTEP,
            #                                                                 state=obs[j],
            #                                                                 target_pos=target_pos[wp_index, :],
            #                                                                 target_rpy=simulation.init_rpys[j, :])
            
            
            print(f"next target: {action[j, :]}")
        simulation.wp_counters += 1
        payload_position = simulation.weight.get_position()
        # print(obs[0][0], obs[0][1], obs[0][2])
        # print(obs)
        if np.all(simulation.wp_counters > len(target_pos)):
            
            break
            # for k in range(p.getNumJoints(simulation.drone_id)):
                
                
            #     # Set the motor to torque control with zero torque to disable it
            #     p.setJointMotorControl2(simulation.drone_id, k, p.TORQUE_CONTROL, force=0)
            #     # p.setJointMotorControl2(env.DRONE_IDS[0], i, p.VELOCITY_CONTROL, targetVelocity=0)
            
            simulation.reset(np.array([0, 0, 0]))
        
        act2 = action[j, :].squeeze()
       
        for j in range(simulation.num_drones):
            if wp_index > args.control_freq_hz*6:
                simulation.logger.log(drone=j,
                            timestamp=i / simulation.control_freq_hz,
                            state=np.hstack([obs[j][0:3],
                                                np.zeros(4),
                                                obs[j][3:15],
                                                act2[j]
                                                ]),
                            control=np.zeros(12),
                            payload_position = payload_position
                            )
            
            
            
        simulation.render()
        if simulation.gui:
            sync(i, start_time, simulation.CTRL_TIMESTEP)
    
    simulation.close()
    simulation.logger.save()
    simulation.logger.save_as_csv("pid")
    if simulation.plot:
        simulation.logger.plot()

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
    parser.add_argument('--control_freq_hz', default=240, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=60, type=int, help='Duration of the simulation in seconds (default: 60)', metavar='')
    parser.add_argument('--output_folder', default='results', type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=False, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--hover_height', default=2.0, type=float, help='Hover height in meters (default: 2.0)', metavar='')
    parser.add_argument('--trajectory_file', default='trajectory_1.csv', type=str, help='CSV file with trajectory points (default: "trajectory_1.csv")', metavar='')
    parser.add_argument('--processed_trajectory_file', default='processed_trajectory_1.csv', type=str, help='CSV file with processed trajectory points (default: "processed_trajectory_1.csv")', metavar='')
    
    args = parser.parse_args()

    # simulation = TetherModelSimulationEnvRPM(np.array([0, 0, 0]),
    #     drone=args.drone,
    #     num_drones=args.num_drones,
    #     physics=args.physics,
    #     gui=args.gui,
    #     record_video=args.record_video,
    #     plot=args.plot,
    #     user_debug_gui=args.user_debug_gui,
    #     obstacles=args.obstacles,
    #     simulation_freq_hz=args.simulation_freq_hz,
    #     control_freq_hz=args.control_freq_hz,
    #     duration_sec=args.duration_sec,
    #     output_folder=args.output_folder,
    #     colab=args.colab,
    #     hover_height=args.hover_height
    # )
    
    start_pos = np.array([1.6248056640625,0,0.16617747497558594])
    simulation = TetherModelSimulationEnvPID(start_pos=start_pos,
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
        # branch_init_pos=[-1.6,-0.004,1.8]
        )


    target_pos = load_and_process_trajectory(args.trajectory_file, args.processed_trajectory_file, args.hover_height, args.control_freq_hz, start_pos)
    
    # target_pos = data[['drone_x', 'drone_y', 'drone_z']].values
    # print(len(target_pos))
    # print(waypoints)
    run_simulation(simulation, target_pos)
