import math
from typing import Tuple
import numpy as np


def interpolate_distance(distance, max_value, max_reward, min_value=0, min_reward=0):
    return min_reward + ((max_reward - min_reward) * (distance - min_value)) / (max_value - min_value)


class CircularApproachingReward():
    down_centre_x, down_centre_y, down_centre_z, down_start, down_end = 0,0, 2.7, 225, 315
    up_centre_x, up_centre_y,up_centre_z, up_start, up_end = 0, 0, 4.5, 45, 135
    radius = 3
    reward_penalty = 5
    has_already_collided = False

    def reward_fun(self, state, has_collided, dist_tether_branch, dist_drone_branch,
                   num_wraps) -> Tuple[float, bool, bool]:

        reward = min(self._calculate_sector_reward(state),
                     self._calc_physical_reward(dist_tether_branch, dist_drone_branch)) + (
                         1.0 if has_collided else 0.0)
        
        reward = self.clip_norm(reward, -3.5, 1.0)
        return reward - 1, False, False

    def clip_norm(self, reward, min_val, max_val):
        clipped_val = min(max_val, max(reward, min_val))
        normalized_val = (clipped_val - min_val) / (max_val - min_val)
        return normalized_val

    def _calculate_sector_reward(self, state):
        x, y, z = state
        #down_center)xyz is the branch position
        is_within, norm_distance = self._within_sector(self.down_centre_x, self.down_centre_y, self.down_centre_z, self.radius,
                                                       self.down_start, self.down_end, x, y, z)
        if is_within:
            return - self.reward_penalty * norm_distance

        is_within, norm_distance = self._within_sector(self.up_centre_x, self.up_centre_y, self.up_centre_z, self.radius,
                                                       self.up_start, self.up_end, x,y, z)
        if is_within:
            return - self.reward_penalty * norm_distance

        return 0.0

    def _within_sector(self, center_x, center_y, center_z, radius, start_angle, end_angle, point_x, point_y,point_z):
        distance = math.sqrt((point_x - center_x) ** 2 + (point_y - center_y) ** 2 +(point_z - center_z) ** 2)
        # distance = math.sqrt((point_x - center_x) ** 2 +(point_z - center_z) ** 2)

        # Check if the point is within the circle's radius
        if distance > radius:
            return False, None

        # # Calculate the angle from the center to the point in radians
        angle_radians = math.atan2(point_z - center_z, point_x - center_x)


    
        # # Calculate the angle in 3D space using the vector from the center to the point
        # vector = np.array([point_x - center_x, point_y - center_y, point_z - center_z])
        # reference_vector = np.array([center_x, center_y, center_z]) 

        # # Calculate the angle between the vector and the reference vector
        # cos_angle = np.dot(vector, reference_vector) / (np.linalg.norm(vector) * np.linalg.norm(reference_vector))
        # angle_radians = math.acos(cos_angle)
        
        
        
        # Convert angle to degrees for easier handling, normalizing to [0, 360)
        angle_degrees = math.degrees(angle_radians) % 360
        start_angle %= 360
        end_angle %= 360
        
        if start_angle > end_angle:
            within = (angle_degrees >= start_angle and angle_degrees <= 360) or (
                angle_degrees >= 0 and angle_degrees <= end_angle)
        else:
            within = (start_angle <= angle_degrees <= end_angle)

        return within, 1 - (distance / radius)

    def _calc_physical_reward(self, dist_tether_branch, dist_drone_branch):
        reward = - dist_tether_branch + self._calc_drone_branch_reward(dist_drone_branch)
        return reward

    def _calc_drone_branch_reward(self, dist_drone_branch: np.ndarray) -> float:
        """
        Calculate reward for drone hitting the branch: Ring based
        - Inner: -10, Outer: 0, Between: 0:-5
        """
        if dist_drone_branch < 0.1:  # A collision
            return -5.0
        elif dist_drone_branch < 0.2:  # Quite close
            return interpolate_distance(dist_drone_branch, 0.1, -5, min_value=0.2)
        else:
            return 0

    def end(self):
        pass
