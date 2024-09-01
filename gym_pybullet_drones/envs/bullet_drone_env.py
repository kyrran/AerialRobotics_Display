import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import os
import pandas as pd
import time

from gym_pybullet_drones.envs.TetherModelSimulationEnvPID import TetherModelSimulationEnvPID
from gym_pybullet_drones.rewards.reward_system import RewardSystem


class BulletDroneEnv(TetherModelSimulationEnvPID):
    """
    BulletDroneEnv now inherits from TetherModelSimulationEnvPID, allowing direct use of PID control and other functionalities.
    """

    # metadata = {"render_modes": ["console", "human"]}
    reset_pos = [2, 0, 3]
    centre_pos = np.array([0.0, 0.0, 3.0])  # Goal state
    reset_pos_distance = 2.0

    def __init__(self, render_mode: str = "human", phase: str = "all", log_dir=None, branch_pos=[0,0,2.7], client = True) -> None:
        super().__init__(start_pos=self._generate_reset_position(42), branch_init_pos=branch_pos, client=client)
        # You can keep or modify the action and observation spaces depending on your specific needs
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = self._actionSpace()
        
        # print(self.action_space)
        self.observation_space = self._observationSpace()
        
        self.render_mode = render_mode
        self.num_steps = 0
        self.should_render = True
        self.reward = RewardSystem(phase)
        self.is_logging = bool(log_dir is not None)
        if self.is_logging:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
            self.csv_file = None
            self.timestep = 0
            self.df = None

    def reset(self, seed: int=None, options: Dict[str, Any] = None,
              degrees: int = None, position=None, branch_pos=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        
        # Reset using the parent class method
        reset_pos = position if position is not None else self._generate_reset_position(seed)
        super().reset(reset_pos, seed, branch_pos=branch_pos)
        
        self.num_steps = 0
        self.reward.reset()

        if self.is_logging:
            self.timestep = 0
            timestamp = int(time.time())
            self.csv_file = os.path.join(self.log_dir, f"log_{timestamp}.csv")
            self.df = pd.DataFrame(columns=["timestep", "x", "y", "z", "roll", "pitch", "yaw", "phase"])
            # pos, orn_euler = self.simulator.drone.get_full_state()
            
            for i in range(self.num_drones):
                drone_state = self._getDroneStateVector(i)
                pos = drone_state[0:3]
                orn_euler = drone_state[7:10]
            
            self.log_state(pos, orn_euler, 0)

        aug_state = np.append(reset_pos, 0.0).astype(np.float32)
        return aug_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        action = np.reshape(action, (self.NUM_DRONES, -1))
        # print(f"in bullet, the action given is: {action}")
        obs, reward, terminated, truncated, info = super().step(action)
        current_position = self.get_drone_currrent_pos()
        num_wraps = info['num_wraps']
        augmented_state = np.append(current_position, num_wraps).astype(np.float32)
                
        if self.is_logging:
            
            for i in range(self.num_drones):
                drone_state = self._getDroneStateVector(i)
                pos = drone_state[0:3]
                orn_euler = drone_state[7:10]
                
                
                # Phase 0 (Approaching if num_wraps <= 0.75), Phase 1 (Otherwise)
                self.log_state(pos, orn_euler, 1 if num_wraps > 0.75 else 0)
                self.timestep += 1
                if terminated:
                    self.save_to_csv()

        return augmented_state, reward, terminated, truncated, info


    def _generate_reset_position(self, seed):
        if seed is not None:
            np.random.seed(seed)
        angle = np.random.uniform(0, np.pi / 3)

        return self._generate_reset_position_from_radians(angle)

    def _generate_reset_position_from_degrees(self, degrees):
        return self._generate_reset_position_from_radians(np.radians(degrees))

    def _generate_reset_position_from_radians(self, radians):
        x_offset = self.reset_pos_distance * np.cos(radians)
        y_offset = max(0, self.reset_pos_distance * np.sin(radians))

        reset_pos = self.centre_pos + np.array([x_offset, 0, y_offset], dtype=np.float32)
        return reset_pos.astype(np.float32)

    def log_state(self, pos, orn_euler, phase):
        # Log state to DataFrame
        self.df = self.df._append({
            "timestep": self.timestep,
            "x": pos[0], "y": pos[1], "z": pos[2],
            "roll": orn_euler[0], "pitch": orn_euler[1], "yaw": orn_euler[2],
            "phase": phase
        }, ignore_index=True)

    def save_to_csv(self):
        self.df.to_csv(self.csv_file, index=False)
        
        
    def calc_reward_and_done(self, state, num_wraps=0.0):
        branch_pos = np.array([0.0, 0.0, 2.7])  # Branch position
        tether_pos = state - np.array([0, 0, 0.5])
        dist_tether_branch = np.linalg.norm(tether_pos - branch_pos)
        dist_drone_branch = np.linalg.norm(state - branch_pos)
        has_collided = bool(dist_tether_branch < 0.1)

        reward = self.reward.calculate(state, has_collided, dist_tether_branch, dist_drone_branch,
                                             num_wraps=num_wraps)
        done = self.reward.refer_terminated()
        return reward, done
    
    
    def calc_reward(self, state, num_wraps=0.0):
        branch_pos = np.array([0.0, 0.0, 2.7])  # Branch position
        tether_pos = state - np.array([0, 0, 0.5])
        dist_tether_branch = np.linalg.norm(tether_pos - branch_pos)
        dist_drone_branch = np.linalg.norm(state - branch_pos)
        has_collided = bool(dist_tether_branch < 0.1)

        reward = self.reward.calculate(state, has_collided, dist_tether_branch, dist_drone_branch,
                                             num_wraps=num_wraps)
        
        return reward
        
    
    