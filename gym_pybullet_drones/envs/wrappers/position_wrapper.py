import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple
import numpy as np


class PositionWrapper(gym.Wrapper):
    MAGNITUDE = 0.005
    MAX_STEP = 0.25
    # MAX = 6
    # MIN = -3
    NUM_ACTIONS_PER_STEP = 25

    def __init__(self, env) -> None:
        super().__init__(env)

        # Position Based Action Space
        self.action_space = env.action_space
        # print(self.action_space)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # TODO: Do this relative to the other environments - make it nicer :)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        low = env.observation_space.low[0][:3]
        high = env.observation_space.high[0][:3]
        
        low = np.append(low, 0)
        high = np.append(high, 1)
        
        
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_state = None
        env.unwrapped.should_render = False
        self.num_steps = 0
        self.NUM_DRONES = env.get_wrapper_attr('NUM_DRONES')
        
        self.previous_action = None
        self.counter = 0
        
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        
        action = np.reshape(action, (self.NUM_DRONES, -1))
        
        self.counter += 1
        # print(f"in position, the action given is: {action}")
        # Execute the step in the environment with the intended action
        state, reward, terminated, truncated, info = self._take_single_step(action)
        
        avg_reward = reward

        # Check if the drone is hovering (consecutive actions are the same)
        # is_hovering = (np.array_equal(action, self.previous_action) and action[0][2] <=2.0)
        # print((np.array_equal(action, self.previous_action)),action[0][2] <=2.0 )
        # is_hovering = action[0][2] <=2.0
        # if not is_hovering:
        # #     # Apply the step penalty only if not hovering
        #     avg_reward -= ((self.num_steps * self.num_steps) / 10_000)
        #     # print(f"deducted:{((self.num_steps * self.num_steps) / 10_000)}")
        #     self.num_steps += 1
        
        # Update the previous action to the current one
        # self.previous_action = action
        # print(f"num of step: {self.num_steps}")
        return state, avg_reward, terminated, truncated, info
    
    
    # def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        
    #     action = np.reshape(action, (self.NUM_DRONES, -1))
        
        
    #     action = action * self.MAX_STEP
    #     self.num_steps += 1

    #     action = action / self.NUM_ACTIONS_PER_STEP
    #     total_reward = 0
    #     actual_steps_taken = 0

    #     for i in range(self.NUM_ACTIONS_PER_STEP):
    #         state, reward, terminated, truncated, info = self._take_single_step(action)
    #         if terminated or truncated:
    #             break
    #         total_reward += reward
    #         actual_steps_taken += 1

    #     avg_reward = total_reward / actual_steps_taken if actual_steps_taken != 0 else 0
    #     avg_reward -= ((self.num_steps * self.num_steps) / 10_000)
    #     return state, avg_reward, terminated, truncated, info
    
    
    
    def reset(self, seed: int = None, options: Dict[Any, Any] = None,
              degrees: int = None, position=None) -> Tuple[np.ndarray, Dict[Any, Any]]:
        state, info = self.env.reset(seed, options, degrees, position)
        self.current_state = state
        self.num_steps = 0
        self.counter = 0
        return state, info

    # def render(self):
    #     print(f'Agent position: {self.current_state}')


    def _is_close_enough(curr_pos, target_pos, threshold=0.001) -> bool:
        distance = np.linalg.norm(curr_pos - target_pos)
        return distance <= threshold

    def _take_single_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        # Calculate direction vector

        state, reward, terminated, truncated, info = self.env.step(action)

        self.current_state = state

        # return state, reward, terminated, truncated or self.num_steps >= 100, info
        return state, reward, terminated, truncated or self.counter >= 300, info