from gym_pybullet_drones.utils.util_graphics import print_green, print_red
from gym_pybullet_drones.utils.util_file import load_json, make_dir
from gym_pybullet_drones.utils.args_parsing import StoreDict
from gym_pybullet_drones.utils.rl.lr_schedular import LinearLearningRateSchedule
import argparse
import numpy as np
import glob
import os
import time
from gym_pybullet_drones.utils.utils import str2bool, sync


from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
# from gym_pybullet_drones.envs.wrappers.two_dim_wrapper import TwoDimWrapper
from gym_pybullet_drones.envs.wrappers.position_wrapper import PositionWrapper
from gym_pybullet_drones.envs.wrappers.symmetric_wrapper import SymmetricWrapper
from gym_pybullet_drones.envs.wrappers.hovering_wrapper import HoveringWrapper
from gym_pybullet_drones.envs.wrappers.custom_monitor import CustomMonitor


from stable_baselines3.common.evaluation import evaluate_policy


DEMO_PATH = "/home/kangle/Documents/FYP/gym-pybullet-drones/gym_pybullet_drones/demonstration/rl_demos_new"
DEFAULT_CHECKPOINT = 5000

# ---------------------------------- RL UTIL ----------------------------------


def generate_graphs(directory, phase="all"):
    from gym_pybullet_drones.utils.plot_graphs import plot_reward_visualisation, read_csv_file
    from gym_pybullet_drones.models.sample_trajectories_from_model import sample_trajectories

    # visualise reward function used
    print_green("Generating Reward Visualisation")
    plot_reward_visualisation(directory, show=False)

    # visualise training rewards
    print_green("Generating Reward Logs")
    read_csv_file(f"{directory}/logs.monitor.csv", show=False)

    # visualise sample trajectories
    print_green("Generating Sample Trajectories")
    sample_trajectories(directory, show=False, phase=phase)


# Shows the demonstration data in the enviornment - useful for verification purpose
def show_in_env(env, transformed_data):
    """Display the demonstration data in the environment."""
    done = False
    start_time = time.time()

    for i, (state, next_obs, action, reward1, done_flag, info) in enumerate(transformed_data):
       
        obs, reward, done, truncated, _ = env.step(action)
        
        # print(f"State: {next_obs}, Simulated State: {obs}")
        # print(f"Reward: {reward1}, Simulated reward: {reward}")
        # print(f"action: {action}")
        
        env.render()

        sync(i, start_time, env.get_wrapper_attr('CTRL_TIMESTEP'))
        
        if done or truncated:
            print("1Episode finished")
            break
        
    while not done and not truncated:
        _, _, done, truncated, _ = env.step(obs[:3])
        if done:
            print("Episode finished")
        if truncated:
            print("Episode Truncated")
            
        



# ----------------------------------- DATA ------------------------------------

def get_buffer_data(env, directory, show_demos_in_env):
    """Load the demonstration data from JSON files and optionally display it in the environment."""
    
    pattern = f"{directory}/rl_demo_new_*.json"
    files = glob.glob(pattern)
    all_data = []
    
    for file in files:
        print(file)
        json_data = load_json(file)
        transformed_data = convert_data(env, json_data)
        
        if show_demos_in_env:
            
            position = transformed_data[0][0][:3]
            # position[2] = 3.0
            env.reset(position=position)
            show_in_env(env, transformed_data)
            
        
        all_data.extend(transformed_data)
    
    return all_data


def convert_data(env, json_data):
    dataset = []
    num = 0
    for item in json_data:
       
        x, y, z, t = item['state']
        obs = np.append(np.array([x, y, z, t]), num / 100.0)
        
        
        _next_obs = item['next_state']
        x, y, z, t = _next_obs
        next_obs = np.append(np.array(np.array([x, y, z, t])), (num + 1) / 300.0)
        
        
        # print(f"before:{np.array(item['action'])}")
        # Normalised action TODO: Define this relative to the env so it's consistent
        # x1,y1,z1 = np.array(item['action'])
        
        # action = np.array([x1-x, y1-y,z1-z])
        action = np.array(item['action'])
        # print(f"after:{action}")
        reward = np.array(item['reward'])
        done = np.array([False])
        info = [{}]
        dataset.append((obs, next_obs, action, reward, done, info))
        num = num + 1
    # last_action = np.array(next_obs)
    
    for _ in range(1):  # Adds an extra action on the end which helps with wrapping.
        dataset.append((next_obs, next_obs, next_obs[:3], reward, done, info))
    dataset.append((next_obs, next_obs, next_obs[:3], reward, np.array([True]), info))
    return dataset

# ---------------------------- ENVIRONMENT & AGENT ----------------------------


def get_checkpointer(should_save, dir_name, checkpoint, phase="all"):
    from gym_pybullet_drones.utils.CheckpointCallback import CheckpointCallback

    if should_save and checkpoint is not None:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint,
            save_path=f"../models/{dir_name}/training_logs/",
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
            phase=phase)
        return checkpoint_callback
    return None


def get_env(dir_name, render_mode, phase):
    
    env = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(render_mode=render_mode, phase=phase, client = False))))
    
    
    # env = (BulletDroneEnv(render_mode=render_mode, phase=phase)))
    if dir_name is not None:
        env = CustomMonitor(env, f"../models/{dir_name}/logs")

    return env


def get_agent(algorithm, env, demo_path, show_demos_in_env, hyperparams):
    from gym_pybullet_drones.algorithms.sacfd import SACfD
    from gym_pybullet_drones.algorithms.dual_buffer import DualReplayBuffer
    from stable_baselines3 import SAC

    _policy = "MlpPolicy"
    _seed = 0
    _batch_size = hyperparams.get("batch_size", 64)
    _policy_kwargs = dict(net_arch=[128, 128, 64])
    _lr_schedular = LinearLearningRateSchedule(hyperparams.get("lr", 0.0002))

    print_green(f"Hyperparamters: seed={_seed}, batch_size={_batch_size}, policy_kwargs={_policy_kwargs}, " + (
                f"lr={_lr_schedular}"))

    if algorithm == "SAC":
        agent = SAC(
            _policy,
            env,
            seed=_seed,
            batch_size=_batch_size,
            learning_rate=_lr_schedular,
            gamma=0.96,
            policy_kwargs=_policy_kwargs,
        )
    elif algorithm == "SACfD":
        agent = SACfD(
            _policy,
            env,
            seed=_seed,
            batch_size=_batch_size,
            policy_kwargs=_policy_kwargs,
            learning_starts=0,
            gamma=0.96,
            learning_rate=_lr_schedular,
            replay_buffer_class=DualReplayBuffer
        )
        pre_train(agent, env, demo_path, show_demos_in_env)

    else:
        print_red("ERROR: Not yet implemented",)
    return agent


def get_existing_agent(existing_agent_path, env):
    from gym_pybullet_drones.algorithms.sacfd import SACfD

    try:
        # Check if the file path ends with .zip
        if not existing_agent_path.endswith('.zip'):
            raise ValueError("The file path must end with .zip")

        # Load the SAC model directly from the zip file
        model = SACfD.load(existing_agent_path)
        model.set_env(env)

        # Check for the replay buffer file in the same directory
        replay_buffer_path = os.path.join(os.path.dirname(existing_agent_path), 'replay_buffer.pkl')
        if os.path.exists(replay_buffer_path):
            print("A replay buffer is being loaded.")
            model.load_replay_buffer(replay_buffer_path)

        print(f"Buffer Size: {model.replay_buffer.size()}")
        return model
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


def pre_train(agent, env, demo_path, show_demos_in_env):
    from stable_baselines3.common.logger import configure
    tmp_path = "/tmp/sb3_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    agent.set_logger(new_logger)

    data = get_buffer_data(env, demo_path, show_demos_in_env)
    # print(data[-1])
    print("Buffer Size in pre_train(): ", agent.replay_buffer.size())

    for i in range(5):
        for obs, next_obs, action, reward, done, info in data:
            # obs = obs[:7]  # Only take the first 4 elements
            # next_obs = next_obs[:7]  # Same for next observation
            # print(f"main action:{action}")
            # print(f"obs, next obs, action:{obs.shape},{next_obs.shape},{action.shape}")
            print(f"Pre-training with obs, action, next state, reward: {obs},{action},{next_obs} reward: {reward}")
            agent.replay_buffer._add_offline(obs, next_obs, action, reward, done, info)
    print("Online Buffer Size: ", agent.replay_buffer.online_replay_buffer.size())
    print("Offline Buffer Size: ", agent.replay_buffer.offline_replay_buffer.size())
    print_green("Pretraining Complete!")


# ----------------------------------- MAIN ------------------------------------

def test_agent(agent, env, num_episodes=5):
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = np.array(obs) 
        start = time.time()
        done = False
        total_reward = 0
        counter = 0
        while not done or not truncated:
             # Ensure obs is in the correct format
            # print(f"Episode {episode + 1}, Step {counter}: Observation shape: {obs}")
            action, _states = agent.predict(obs, deterministic=True)
            # print(f"Testing, action: {action}")
            obs, reward, done, truncated, info = env.step(action)
            
            payload_position = env.weight.get_position()
            
            total_reward += reward
            env.render()
            sync(counter, start, env.CTRL_TIMESTEP)
            if done or truncated:
                print(f"hi, obs:{obs.shape}")
                break
            counter += 1
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
        
        # obs2 = obs.squeeze()
        # act2 = action.squeeze()
        
        # env.logger.log(drone=0,
        #             timestamp=episode/env.CTRL_FREQ,
        #             state=np.hstack([obs2[0:3],
        #                                 np.zeros(4),
        #                                 obs2[3:15],
        #                                 act2
        #                                 ]),
        #             control=np.zeros(12),
        #             payload_position = payload_position
        #             )
    env.close()
    
    # env.logger.save()
    # env.logger.save_as_csv("pid")
    # env.logger.plot()

def main(algorithm, timesteps, filename, render_mode, demo_path, should_show_demo, checkpoint, hyperparams,
         existing_agent, save_replay_buffer, phase):
    save_data = filename is not None
    dir_name = make_dir(filename)

    env = get_env(dir_name, render_mode, phase)
    
     #### Check the environment's spaces ########################
    print('[INFO] Action space:', env.action_space)
    print('[INFO] Observation space:', env.observation_space)

    checkpoint_callback = get_checkpointer(save_data, dir_name, checkpoint, phase)
    
    if existing_agent is None:        
        agent = get_agent(algorithm, env, demo_path, should_show_demo, hyperparams)

    else:
        agent = get_existing_agent(existing_agent, env)

    agent.learn(timesteps, log_interval=10, progress_bar=True, callback=checkpoint_callback)

    print_green("TRAINING COMPLETE!")

    if save_data:
        agent.save(f"../models/{dir_name}/model")
        if save_replay_buffer:
            agent.save_replay_buffer(f"../models/{dir_name}/replay_buffer")
        print_green("Model Saved")
    
    env.close()
    
    test_env = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(render_mode=render_mode, phase=phase, client = False))))
    
    test_env_nogui = HoveringWrapper(PositionWrapper(SymmetricWrapper(BulletDroneEnv(render_mode=render_mode, phase=phase, client = True))))
    
    mean_reward, std_reward = evaluate_policy(agent,
                                              test_env_nogui,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    
    
      # Testing
    # test_env = get_env(dir_name, render_mode, phase)  # Create a new testing environment
    test_agent(agent, test_env)
    
    
    

    if save_data:
        generate_graphs(directory=f"../models/{dir_name}", phase=phase)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Training for Tethered Drone Perching")

    # Number of timesteps
    parser.add_argument('-t', '--timesteps', type=int, required=True,
                        help='Number of timesteps for training (e.g., 40000)')

    # Choice of algorithm
    parser.add_argument('-algo', '--algorithm', type=str, choices=['SAC', 'SACfD'], required=True,
                        help='Choice of algorithm: SAC or SACfD')

    # Output filename for logs
    parser.add_argument('-o', '--output-filename', type=str, default=None, help='Filename for storing logs')
    parser.add_argument('--save-replay-buffer', action='store_true', help='Saves the replay model from the buffer.')

    # Graphical user interface
    parser.add_argument('-gui', '--gui', action='store_true', help='Enable graphical user interface')

    # Demonstration path
    parser.add_argument('--demo-path', type=str, default=DEMO_PATH,
                        help=f"Path to demonstration files (default: {DEMO_PATH}")

    # Show demonstrations in visual environment
    parser.add_argument('--show-demo', action='store_true', help='Show demonstrations in visual environment')

    # Checkpoint episodes
    parser.add_argument('--checkpoint-episodes', type=int, default=DEFAULT_CHECKPOINT,
                        help='Frequency of checkpoint episodes (default: 5000)')
    parser.add_argument('--no-checkpoint', action='store_true', help='Perform NO checkpointing during training.')

    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict,
                        help="Overwrite hyperparameter (e.g. lr:0.01 batch_size:10)",)

    # Continue Training
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training",
                        default=None, type=str, required=False)

    # Train particular phase
    parser.add_argument("-p", "--phase", type=str, choices=["approaching", "all"], default="all",
                        help="Train a particular phase of a the system.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    algorithm = args.algorithm
    timesteps = args.timesteps
    filename = args.output_filename
    render_mode = "human" if args.gui else "console"
    demo_path = args.demo_path
    should_show_demo = args.show_demo
    checkpoint = args.checkpoint_episodes
    if args.no_checkpoint:
        checkpoint = None
    trained_agent = args.trained_agent
    save_replay_buffer = args.save_replay_buffer
    phase = args.phase

    if algorithm != "SACfD" and demo_path is not None:
        print_red("WARNING: Demo path provided will NOT be used by this algorithm!")

    print_green(f"Algorithm: {algorithm}")
    print_green(f"Timesteps: {timesteps}")
    print_green(f"Render Mode: {render_mode}")
    if filename is None:
        print_red("WARNING: No output or logs will be generated, the model will not be saved!")
    else:
        print_green(f"File Name: {filename}")

    if algorithm == "SACfD":
        print_green(f"Demo Path: {demo_path}")
    print_green(f"Checkpointing: {checkpoint}")

    if trained_agent is not None:
        print_green(f"Using pre-trained agent: {trained_agent}")

    accpetable_hp = ["lr", "batch_size"]
    hyperparams = args.hyperparams if args.hyperparams is not None else dict()
    for key, val in hyperparams.items():
        if key in accpetable_hp:
            print_green(f"\t{key}: {val}")
        else:
            print_red(f"\nUnknown Hyperparameter: {key}")

    main(algorithm, timesteps, filename, render_mode, demo_path, should_show_demo, checkpoint, hyperparams,
         trained_agent, save_replay_buffer, phase)
