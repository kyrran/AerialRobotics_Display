import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob
import numpy as np

limits = {
    'x': (-3, 3),
    'y': (-3, 3),
    'z': (0, 6),
    'roll': (-1, 1),
    'pitch': (-1, 1),
    'yaw': (-1, 1)}


# Function to read the CSV file and plot each column over time
def plot_columns_over_time(input_file, output_dir):
    # Read the CSV file
    data = pd.read_csv(input_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    first_phase_one_index = data[data['phase'] == 1].index[0] if 1 in data['phase'].values else None

    def plot_data(sub_data, suffix):
        # Plot each column over time
        for column, (min_limit, max_limit) in limits.items():
            plt.figure(figsize=(10, 6))
            plt.plot(sub_data['timestep'], sub_data[column], label=f'{column} over time', color='blue')

            actual_min = min(sub_data[column].min(), min_limit)
            actual_max = max(sub_data[column].max(), max_limit)
            plt.ylim(actual_min, actual_max)

            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel(column)
            plt.title(f'{column} over Time')
            plt.grid(True)

            # Save the plot
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f'{base_filename}_{column}_over_time_{suffix}.png')
            plt.savefig(output_file)
            plt.close()

    def plot_xz_with_roll(data, filename_suffix):
        plt.figure(figsize=(10, 6))
        plt.plot(data["x"], data["z"], color="blue")
        plt.ylim(1.5, 4.5)
        plt.xlim(-1.5, 1.5)

        for i in range(0, len(data), max(1, len(data) // 20)):
            x = data["x"].iloc[i]
            z = data["z"].iloc[i]
            roll = data["roll"].iloc[i]

            line_length = 0.05
            dx = line_length * np.cos(roll)
            dz = line_length * np.sin(roll)

            plt.plot([x - dx, x + dx], [z - dz, z + dz], color='red')

        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('x vs z with roll')
        plt.legend()
        plt.grid(True)

        # Save the plot
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f'{base_filename}_xz_with_roll_{filename_suffix}.png')
        plt.savefig(output_file)
        plt.close()

    plot_data(data, 'full')

    if first_phase_one_index is not None:
        plot_data(data.iloc[:first_phase_one_index], "phase0")

        plot_data(data.iloc[first_phase_one_index:], "phase1")

        plot_xz_with_roll(data.iloc[first_phase_one_index:], "phase1")


def plot_reward_visualisation(directory, show=True, plot_type=0.0):
    from gym_pybullet_drones.envs.bullet_drone_env import BulletDroneEnv
    env = BulletDroneEnv(render_mode="console")

    # Set the range for x and z
    x_values = np.linspace(-3, 3, 100)
    z_values = np.linspace(0, 6, 100)

    # Create a grid of x, y=0, z values
    x_grid, z_grid = np.meshgrid(x_values, z_values)

    # Compute the rewards for each position
    rewards = np.array([[env.calc_reward([x, 0, z], plot_type) for x, z in zip(x_row, z_row)]
                        for x_row, z_row in zip(x_grid, z_grid)])

    # Branch coordinates
    branch_x = 0  # Example x-coordinate
    branch_z = 2.7  # Example z-coordinate

    # Plotting
    plt.figure(figsize=(10, 6))
    heatmap = plt.imshow(rewards, extent=[-3, 3, 0, 6], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(heatmap, label='Reward')
    plt.title('Reward Function Visualization')
    plt.xlabel('X coordinate')
    plt.ylabel('Z coordinate')

    # Add the branch point
    plt.scatter(branch_x, branch_z, color='red', label='Branch', s=20)  # 's' adjusts the size of the point
    plt.legend()

    suffix = "approaching" if plot_type == 0 else "wrapping"

    if directory is not None:
        plt.savefig(f"{directory}/reward_visualisation_{suffix}.png")
    if show:
        plt.show()
    else:
        plt.clf()


def read_csv_file(filename, num_episodes=None, smoothing=10, show=True):
    from utils.graphics.plot_rl_rewards_training import plot_rl_reward_graph
    try:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(filename, comment='#', header=0)

        directory = os.path.dirname(filename)
        output_filename = os.path.join(directory, 'rewards_graph.png')

        rewards = data['r']
        lengths = data['l']
        if num_episodes is not None:
            rewards = rewards.iloc[0:num_episodes]
            lengths = lengths.iloc[0:num_episodes]
        plot_rl_reward_graph(rewards, episode_lens=lengths, output_filename=output_filename, window_size=smoothing,
                             show_plot=show)
    except FileNotFoundError:
        print("Error: File not found. Please check the filename and try again.")
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
    except pd.errors.ParserError:
        print("Error: File could not be parsed. Check if the file is a valid CSV.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description='Main Parser', add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')

    subparsers = parser.add_subparsers(dest="command")

    # Position Plots Parsing
    parser_plots = subparsers.add_parser('plots', help="Create positional plots for all dimensions from logs")
    parser_plots.add_argument('-o', '--output', required=True, help='Directory to save the output plots')
    group_plots = parser_plots.add_mutually_exclusive_group(required=True)
    group_plots.add_argument('-i', '--input', help='Path to the input CSV file')
    group_plots.add_argument('-d', '--directory', help='Path to the directory containing CSV files')

    # Rewards Visualisation Based Parser
    parser_rewards = subparsers.add_parser('rewards', help='Create rewards visualizations')
    parser_rewards.add_argument('-o', '--output_directory', default=None, help='Output directory for rewards')
    parser_rewards.add_argument('-p', '--plot_type', choices=[0, 1], type=int, required=True, help='Plot type (0 or 1)')

    # Learning Visualisation Based Parser
    parser_learning = subparsers.add_parser('learn', help='Create reward learning graph')
    parser_learning.add_argument('-i', '--input', required=True, help='Path to the learning input CSV file')
    parser_learning.add_argument('-n', '--num-episodes', default=None, type=int,
                                 help='Maximum number of episodes to show')
    parser_learning.add_argument('-s', '--smoothing', default=10, type=int, help="Smoothing applied to reward curve.")

    args = parser.parse_args()

    def print_combined_help():
        parser.print_help()
        print("\nSubcommand 'plots' help:")
        parser_plots.print_help()
        print("\nSubcommand 'rewards' help:")
        parser_rewards.print_help()
        print("\nSubcommand 'learn' help:")
        parser_learning.print_help()

    if args.help:
        print_combined_help()
    elif args.command == 'plots':
        if args.input:
            plot_columns_over_time(args.input, args.output)
        elif args.directory:
            csv_files = glob.glob(os.path.join(args.directory, '*.csv'))
            for csv_file in csv_files:
                plot_columns_over_time(csv_file, args.output)

    elif args.command == "rewards":
        plot_reward_visualisation(args.output_directory, True, args.plot_type)

    elif args.command == "learn":
        read_csv_file(args.input, num_episodes=args.num_episodes, smoothing=args.smoothing, show=True)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
