import matplotlib.pyplot as plt
import pandas as pd


def plot_rl_comparison(rewards_1, rewards_2, output_filename=None, window_size=10,
                       title='Comparison of Running Rewards Over Training', show_plot=True):
    """
    Plot a graph of rewards with running average and variance.

    Parameters:
        rewards_1 (pd.Series): A Pandas Series containing rewards.
        rewards_2 (pd.Series): A Pandas Series containing rewards.
        output_filename (str): The filename to save the plot. If None, plot will be displayed but not saved.
        window_size (int): The window size for calculating running average and variance.
        title (str): The title of the plot.
        show_plot (bool): Whether to display the plot or not.

    Returns:
        None
    """
    # Input validation
    if not isinstance(rewards_1, pd.Series):
        raise ValueError("rewards_1 must be a Pandas Series.")
    if not isinstance(rewards_2, pd.Series):
        raise ValueError("rewards_2 must be a Pandas Series.")
    if not isinstance(window_size, int):
        raise ValueError("window_size must be an integer.")
    if window_size < 1:
        raise ValueError("window_size must be greater than or equal to 1.")
    if not isinstance(title, str):
        raise ValueError("title must be a string.")
    if not isinstance(show_plot, bool):
        raise ValueError("show_plot must be a boolean.")
    _plot_reward_comparison(rewards_1, rewards_2, output_filename, window_size, title, show_plot)


def _plot_reward_comparison(rewards_1, rewards_2, output_filename, window_size, title, show_plot):
    # Calculate rolling statistics for rewards
    running_rewards_1 = rewards_1.rolling(window=window_size, min_periods=1).mean()
    running_rewards_2 = rewards_2.rolling(window=window_size, min_periods=1).mean()

    # Create the plot with size and primary y-axis for rewards
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(rewards_1.index, running_rewards_1, color='red', linestyle='-', linewidth=2, label='Ours')
    ax1.plot(rewards_2.index, running_rewards_2, color='blue', linestyle='-', linewidth=2, label='SAC')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward Value', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)

    # Add titles and legends
    plt.title(title)
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')

    # Save the plot to a file if filename is provided
    if output_filename:
        plt.savefig(output_filename)
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.clf()


def plot_rl_reward_graph(rewards, episode_lens=None, output_filename=None, window_size=10,
                         title='Running Rewards and Variance Over Training', show_plot=True):
    """
    Plot a graph of rewards with running average and variance.

    Parameters:
        rewards (pd.Series): A Pandas Series containing rewards.
        output_filename (str): The filename to save the plot. If None, plot will be displayed but not saved.
        window_size (int): The window size for calculating running average and variance.
        title (str): The title of the plot.
        show_plot (bool): Whether to display the plot or not.

    Returns:
        None
    """

    # Input validation
    if not isinstance(rewards, pd.Series):
        raise ValueError("rewards must be a Pandas Series.")
    if episode_lens is not None and not isinstance(episode_lens, pd.Series):
        raise ValueError("episode_lens must be a Pandas Series.")
    if not isinstance(window_size, int):
        raise ValueError("window_size must be an integer.")
    if window_size < 1:
        raise ValueError("window_size must be greater than or equal to 1.")
    if not isinstance(title, str):
        raise ValueError("title must be a string.")
    if not isinstance(show_plot, bool):
        raise ValueError("show_plot must be a boolean.")

    _plot_reward_graph(rewards, episode_lens, output_filename=output_filename, window_size=window_size,
                       title=title, show_plot=show_plot)


def _plot_reward_graph(rewards, episode_lens, output_filename, window_size, title, show_plot):
    # Calculate rolling statistics for rewards
    running_avg = rewards.rolling(window=window_size, min_periods=1).mean()
    running_std = rewards.rolling(window=window_size, min_periods=1).std()

    # Create the plot with size and primary y-axis for rewards
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(rewards.index, running_avg, color='red', linestyle='-', linewidth=2, label='Reward Running Average')
    ax1.fill_between(rewards.index, running_avg + running_std, running_avg - running_std,
                     color='gray', alpha=0.3, label='Reward Variance')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward Value', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)
    ax1.set_ylim(min(-350, rewards.min()), max(0, rewards.max()))

    # Create a second y-axis for episode lengths
    if episode_lens is not None:
        ax2 = ax1.twinx()
        running_avg_episode_lens = episode_lens.rolling(window=100, min_periods=1).mean()
        ax2.plot(running_avg_episode_lens.index, running_avg_episode_lens, color='blue', linestyle='-',
                 linewidth=2, label='Episode Length Average')
        ax2.set_ylabel('Episode Length', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 100)

    # Add titles and legends
    plt.title(title)
    lines, labels = ax1.get_legend_handles_labels()
    if episode_lens is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.legend(lines, labels, loc='upper left')

    # Save the plot to a file if filename is provided
    if output_filename:
        plt.savefig(output_filename)
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.clf()
