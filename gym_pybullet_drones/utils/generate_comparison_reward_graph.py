import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.graphics.plot_rl_rewards_training import plot_rl_comparison


def read_csv_file(filename_1, filename_2, num_episodes=None, show=True):
    try:
        # Load the CSV file into a DataFrame
        directory = os.path.dirname(filename_1)
        output_filename = os.path.join(directory, 'rewards_graph_comparison.png')

        rewards_1, lengths_1 = get_rewards_lengths_from_csv(filename_1, num_episodes)
        rewards_2, lengths_2 = get_rewards_lengths_from_csv(filename_2, num_episodes)

        plot_rl_comparison(rewards_1, rewards_2, output_filename=output_filename, show_plot=show)
    except FileNotFoundError:
        print("Error: File not found. Please check the filename and try again.")
    except pd.errors.EmptyDataError:
        print("Error: File is empty.")
    except pd.errors.ParserError:
        print("Error: File could not be parsed. Check if the file is a valid CSV.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_rewards_lengths_from_csv(filename, num_episodes):
    data = pd.read_csv(filename, comment='#', header=0)

    rewards = data['r']
    lengths = data['l']
    if num_episodes is not None:
        rewards = rewards.iloc[0:num_episodes]
        lengths = lengths.iloc[0:num_episodes]

    return rewards, lengths


if __name__ == "__main__":
    # Check if the filename is given as a command-line argument
    if len(sys.argv) == 3:
        filename_1 = sys.argv[1]
        filename_2 = sys.argv[2]
        read_csv_file(filename_1, filename_2, show=True)
    elif len(sys.argv) == 4:
        filename_1 = sys.argv[1]
        filename_2 = sys.argv[2]
        num_episodes = int(sys.argv[3])
        read_csv_file(filename_1, filename_2, num_episodes=num_episodes, show=True)
    else:
        print("Usage: python generate_reward_graph_from_logs.py <filename> <filename> <num>")
