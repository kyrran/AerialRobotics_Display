import os
import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from gym_pybullet_drones.envs.BaseAviary import BaseAviary

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.envs.simulationComponents.tether import Tether
from gym_pybullet_drones.envs.simulationComponents.weight import Weight
from gym_pybullet_drones.envs.simulationComponents.branch import Branch
from gym_pybullet_drones.envs.simulationComponents.room import Room

from gym_pybullet_drones.rewards.reward_system import RewardSystem


class TetherModelSimulationEnvPID(BaseAviary):
    """Drone simulation class inheriting from BaseRLAviary."""

    def __init__(self, 
                 start_pos=[0, 0, 0],
                 drone=DroneModel.CF2X, 
                 num_drones=1, 
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB, 
                 gui=True, 
                 record_video=False, 
                 plot=True, 
                 user_debug_gui=False, 
                 obstacles=False, 
                 simulation_freq_hz=240, 
                 control_freq_hz=120, 
                 duration_sec=48, 
                 output_folder='results', 
                 colab=False, 
                 hover_height=2.0,
                 obs=ObservationType.KIN, 
                 act=ActionType.PID,
                 branch_init_pos = [0,0,2.7],
                 client = False):
        
        self.drone_init_pos = start_pos
        self.branch_init_pos = branch_init_pos
        self.INIT_XYZS = np.array([self.drone_init_pos for _ in range(num_drones)])
        # self.init_rpys = np.array([[0, 0, i * (np.pi / 2) / num_drones] for i in range(num_drones)])
        self.wp_counters = np.array([0 for _ in range(num_drones)])
        
        
        self.drone = drone
        self.num_drones = num_drones
        self.physics = physics
        self.gui = gui
        self.record_video = record_video
        self.plot = plot
        self.user_debug_gui = user_debug_gui
        self.obstacles = obstacles
        self.simulation_freq_hz = simulation_freq_hz
        self.control_freq_hz = control_freq_hz
        self.duration_sec = duration_sec
        self.output_folder = output_folder
        self.colab = colab
        self.hover_height = hover_height
        self.INIT_RPYS = np.zeros((self.num_drones, 3))
     
        #### Create a buffer for the last .5 sec of actions ########
        self.ACTION_BUFFER_SIZE = int(control_freq_hz//2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
       
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
    
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
            
        
        # Call the parent class constructor with the necessary parameters
        # Call the parent class constructor with the necessary parameters
        super().__init__(drone_model=drone,
                         num_drones=num_drones,
                         neighbourhood_radius=np.inf,
                         initial_xyzs=self.INIT_XYZS,
                         initial_rpys=self.INIT_RPYS,
                         physics=physics,
                         pyb_freq=simulation_freq_hz,
                         ctrl_freq=control_freq_hz,
                         gui=gui,
                         record=record_video,
                         obstacles=False, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False,
                         client = client
                        )
        
    
        

        self.logger = Logger(logging_freq_hz=control_freq_hz,
                             num_drones=num_drones,
                             output_folder=output_folder,
                             colab=colab)

        self.ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]
        
        self.drone_id = self.getDroneIds()
        self.branch = None
        self.weight = None
        self.tether = None
        
        self.CTRL_TIMESTEP = 1.0 / self.control_freq_hz
        
        # Initialize custom simulation components like tether, branch, etc.
        self.initialize_simulation_components(self.branch_init_pos)
        
        
        self.reward_system = RewardSystem("all")
        self.EPISODE_LEN_SEC = 15
        
        self.reward_info = 0.0
        self.dist_drone_branch = 0.0
        self.num_rotations = 0.0
        
        
        
        self.weight_prev_angle = None
        self.drone_prev_angle = None
        self.weight_cumulative_angle_change = 0.0
        self.weight_wraps = 0.0
        self.drone_cumulative_angle_change = 0.0
        self.drone_wraps = 0.0
        self.time = 0
        
        self.max_steps = 100
        
        self.steps = 0
        
        self.step_size =1
        
        self.TARGET_POS = None
       
    ################################################################################


    def initialize_simulation_components(self, branch_pos=None):
        self.branch = Branch()
        if branch_pos is not None:
            self.branch.add_tree_branch(position=branch_pos)
        else:
            self.branch.add_tree_branch(position=self.branch_init_pos)
            
        self.branch_body_id = self.branch.get_body_id()  # Store the branch body ID
        
        
        self.tether = Tether(self.getDroneIds(), length=1.0, drone_position=self.drone_init_pos, physics_client=self.CLIENT)
        self.tether.attach_to_drone(self.getDroneIds())
        payload_start_position_top = self.tether.get_world_centre_bottom()
        self.weight = Weight(payload_start_position_top)
        self.tether.attach_weight(self.weight)
        
        self.TARGET_POS = np.array([0,0,3.2])
        print(self.TARGET_POS)
    def check_collisions(self):
        for part_id in self.tether.get_segments():
            contacts = p.getContactPoints(bodyA=self.branch_body_id, bodyB=part_id)
            if contacts:
                return True
        return False

    def reset(self,pos=None,seed=None, branch_pos=None):
        if pos is not None:
            self.drone_init_pos = pos
        else:
            self.drone_init_pos = np.array([0.0,0.0,0.0])
        
        self.INIT_XYZS = np.array([self.drone_init_pos for _ in range(self.num_drones)])
        self.wp_counters = np.array([0 for _ in range(self.num_drones)])
        self.INIT_RPYS = np.zeros((self.num_drones, 3))
        
        self.reward_system.reset()
        self.step_size = 1
        super().reset()
        self.steps = 0
        self.initialize_simulation_components(branch_pos = branch_pos)
    
    
    def step(self, action):
        
        action = np.reshape(action, (self.NUM_DRONES, -1))
        
        
        # action = action + self.get_drone_currrent_pos()
        print(f"in pid, the action given is: {action}")
        
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
    
    def _distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE==ActionType.PID:
            size = 3
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        # act_lower_bound = np.array([-10*np.ones(size) for i in range(self.NUM_DRONES)])
        # act_upper_bound = np.array([+10*np.ones(size) for i in range(self.NUM_DRONES)])
        
        act_lower_bound = np.array([[-5.25,-2.75, 0] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([ [5.25,2.75, 6.2] for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        # print(act_lower_bound)
        print(f"pid:{spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)}")
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        
        
        rpm = np.zeros((self.NUM_DRONES,4))
        
        
        for k in range(action.shape[0]):
            target = action[k, :] 
            # print(f"{target} = {action[k, :] } + {self.get_drone_currrent_pos()}")
            # print(f"target before process: {target}")
            if self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                # print(f"get state vector: {state[0:3]}")
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=self.step_size,
                    )
                # print(f"pid:{next_pos}")
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add action buffer to observation space ################
            # act_lo = -10
            # act_hi = +10
            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE==ActionType.PID:
                    # obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    # obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
                    
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[-5.25,-2.75,0] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[5.25,2.75, 6.2]for i in range(self.NUM_DRONES)])])
            
            # print(spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32))
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            obs_12 = np.zeros((self.NUM_DRONES,12))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            # print(len(ret))
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
            
    def get_drone_currrent_pos(self):
        self.drone_cur_pos = self._getDroneStateVector(0)[0:3]
        return self.drone_cur_pos
            
    ################################################################################

    def _computeReward(self):
        
        # state = self._getDroneStateVector(0)
        # ret = -np.linalg.norm(self.TARGET_POS-state[0:3])
        
        # num_wraps = self.compute_total_rotation()
        # print(f"reward in pid:{ret}")
        # return ret + num_wraps
        
        """Computes the reward based on the current state and action."""
        
        state = self.get_drone_currrent_pos()
        
        has_collided = self.check_collisions()
        dist_tether_branch = self._distance(self.tether.get_mid_point(), self.branch.get_tree_branch_midpoint())
        dist_drone_branch = self._distance(state, self.branch.get_tree_branch_midpoint())
        num_rotations = self.compute_total_rotation() if has_collided else 0.0
        
        reward = self.reward_system.calculate(state, has_collided, dist_tether_branch, dist_drone_branch, num_rotations)
        self.dist_drone_branch = dist_drone_branch
        self.num_rotations = num_rotations
        self.reward_info = reward
        
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        # return False
        # state = self._getDroneStateVector(0)
        # if np.linalg.norm(self.TARGET_POS-state[0:3]) < .0001:
        #     return True
        # else:
        #     return False
        
        # done = self.current_waypoint_index >= len(self.waypoints) or self.steps >= self.max_steps
        # done = self.steps >= self.max_steps
        done = self.reward_system.refer_terminated()
        return done
    
    ################################################################################
    
    def _computeTruncated(self):
        
        return False
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        state = self._getDroneStateVector(0)
        # if (abs(state[0]) > 1.5 or abs(state[1]) > 1.5 or state[2] > 2.0 # Truncate when the drone is too far away
        #      or abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted
        # ):
        #     return True
        
        # if (abs(state[7]) > .4 or abs(state[8]) > .4 # Truncate when the drone is too tilted - raw and yaw
        # ):
        #     return True
        
        
        # if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
        #     return True
        # else:
        #     return False
        
        
        # Check for extreme tilt (crash due to flipping over)
        # roll, pitch = state[7], state[8]
        # if abs(roll) > np.radians(120) or abs(pitch) > np.radians(120):  # 60 degrees as a crash threshold
        #     print("Crash detected due to extreme tilt!")
        #     return True

        # Check for low altitude (crash into the ground)
        altitude = state[2]
        if altitude < 0.0:  # Assuming 0.1 meters is the threshold for ground impact
            print("Crash detected due to ground impact!")
            return True

        # Check for sudden, large changes in velocity (high impact crash)
        velocity = state[10:13]
        if np.linalg.norm(velocity) > 10:  # Threshold for crash due to high-speed impact
            print("Crash detected due to high-speed impact!")
            return True
        
        if self.steps >= self.max_steps:
            print("Too long episode!")
            return True
        
        
        

    ################################################################################
    
    def _computeInfo(self):
        info = {"distance_to_goal": -self.reward_info, "has_crashed": bool(self.dist_drone_branch < 0.1), "num_wraps": self.num_rotations}

        return info #### Calculated by the Deep Thought supercomputer in 7.5M years
    

    ################################################################################
    
    def compute_total_rotation(self):
        # ANGLE FOR WEIGHT
        (weight_x, _, weight_y), _ = p.getBasePositionAndOrientation(self.tether.segments[-1])
        weight_delta_x = weight_x - 0
        weight_delta_y = 2.7 - weight_y

        # Compute the angle using arctan2, which considers quadrant location
        weight_angle_radians = np.arctan2(weight_delta_x, weight_delta_y)
        weight_angle_degrees = np.degrees(weight_angle_radians)

        if self.weight_prev_angle is not None:
            # Calculate angle change considering the wrap around at 180/-180
            weight_angle_change = weight_angle_degrees - self.weight_prev_angle
            if weight_angle_change > 180:
                weight_angle_change -= 360
            elif weight_angle_change < -180:
                weight_angle_change += 360

            # Update cumulative angle change
            self.weight_cumulative_angle_change += weight_angle_change

            # Update wraps as a float
            self.weight_wraps = self.weight_cumulative_angle_change / 360.0

        # Update the previous angle for the next call
        self.weight_prev_angle = weight_angle_degrees

        # ANGLE FOR DRONE
        (drone_x, _, drone_y) = self.get_drone_currrent_pos()
        
        
        
        drone_delta_x = drone_x - 0
        drone_delta_y = 2.7 - drone_y

        # Compute the angle using arctan2, which considers quadrant location
        drone_angle_radians = np.arctan2(drone_delta_x, drone_delta_y)
        drone_angle_degrees = np.degrees(drone_angle_radians)

        if self.drone_prev_angle is not None:
            # Calculate angle change considering the wrap around at 180/-180
            drone_angle_change = drone_angle_degrees - self.drone_prev_angle
            if drone_angle_change > 180:
                drone_angle_change -= 360
            elif drone_angle_change < -180:
                drone_angle_change += 360

            # Update cumulative angle change
            self.drone_cumulative_angle_change += drone_angle_change

            # Update wraps as a float
            self.drone_wraps = self.drone_cumulative_angle_change / 360.0

        # Update the previous angle for the next call
        self.drone_prev_angle = drone_angle_degrees

        self.time += 1
        return abs(self.weight_wraps)
