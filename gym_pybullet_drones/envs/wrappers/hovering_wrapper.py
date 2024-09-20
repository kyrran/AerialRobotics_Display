import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HoveringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assuming the original observation space is a Box of some shape
        low = np.append(env.observation_space.low, 0)
        high = np.append(env.observation_space.high, 1)

        # Update observation space to include one additional integer for the hover count
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space=env.observation_space
        # print(self.observation_space)
        
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # TODO: Do this relative to the other environments - make it nicer :)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        # Initialize hover count
        
        self.episode_count = 0
        # print(self.observation_space.shape)
        self.NUM_DRONES = env.get_wrapper_attr('NUM_DRONES')

    def reset(self, seed=None, options=None, degrees=None, position=None, branch_pos = None):
        # Reset the environment and hover count
        obs, info = self.env.reset(seed, options, degrees, position,branch_pos)
        self.episode_count = 0
        # Return the augmented observation
        augmented_obs = np.append(obs, self.get_hover_ratio()).astype(np.float32)
        return augmented_obs, info

    def get_hover_ratio(self):
        return min(self.episode_count, 300.0) / 300.0

    def step(self, action, num_wraps=None):
        
        action = np.reshape(action, (self.NUM_DRONES, -1))
        # print(f"hovering wrapper action given:{action}")
        
        # print(f"in hover, the action given is: {action}")
        # Take a step using the underlying environment
        if num_wraps is None:
            obs, reward, done, truncated, info = self.env.step(action)
        else:
            obs, reward, done, truncated, info = self.env.step(action, num_wraps)
        # print(obs.shape)
        # Update the hover count if the action size is less than 0.1
        # is_small_action = np.linalg.norm(action)
        # if self.hover_count > 1 and is_small_action: # Already hovering and we are carrying on staying still
        #     reward = reward
        # else:
        #     reward = reward - 0.2

        # if is_small_action < 0.1:
        self.episode_count += 1
        # print(self.hover_count, np.linalg.norm(action))
        # Augment the observation with the hover count

        augmented_obs = np.append(obs, self.get_hover_ratio()).astype(np.float32)
        # print(f"hovering wrapper obs:{augmented_obs}")
        # print(augmented_obs.shape)
        return augmented_obs, reward, done, truncated, info
