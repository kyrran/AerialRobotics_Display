import matplotlib.pyplot as plt


def plot_trajectories_with_rewards(trajectories, traj_rewards, output_filename=None, window_size=10,
                                   title='Sample Trajectories', show_plot=True):
    # Flatten rewards for normalization
    all_rewards = [reward for rewards in traj_rewards for reward in rewards]

    # Normalize the rewards for colormap
    norm = plt.Normalize(min(all_rewards), max(all_rewards))
    cmap = plt.cm.viridis  # You can choose another colormap if you prefer

    fig, ax = plt.subplots()  # Create a figure and an axes object

    for trajectory, rewards in zip(trajectories, traj_rewards):
        x_values = [state[0] for state in trajectory]
        y_values = [state[1] for state in trajectory]

        for (x, y), reward in zip(zip(x_values, y_values), rewards):
            color = cmap(norm(reward))
            ax.plot(x, y, marker='o', linestyle='-', color=color)

    # Add labels and title
    ax.set_xlabel('X position')
    ax.set_ylabel('Z position')
    ax.set_title(title)

    # Set axis limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 6)

    # Add a grid
    ax.grid(True)

    # Highlight the center point (0, 3)
    ax.scatter([0], [3], color='red', zorder=5)  # Zorder for making the point appear on top of the line
    ax.annotate('Branch', xy=(0, 3), xytext=(0, -15), textcoords='offset points', ha='center', color='red')

    # Create a ScalarMappable object for the color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for matplotlib < 3.1

    # Add the color bar to the figure
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Rewards')

    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()
    else:
        plt.clf()


def plot_trajectories(trajectories, output_filename=None, window_size=10,
                      title='Sample Trajectories', show_plot=True):
    for trajectory in trajectories:
        x_values = [state[0] for state in trajectory]
        y_values = [state[1] for state in trajectory]
        plt.plot(x_values, y_values, marker='o', linestyle='-', label='label', color='blue')

    # Add labels and title
    plt.xlabel('X position')
    plt.ylabel('Z position')
    plt.title(title)

    # Set axis limits
    plt.xlim(-3, 3)
    plt.ylim(0, 6)

    # Add a grid
    plt.grid(True)

    # Highlight the center point (0, 3)
    plt.scatter([0], [3], color='red', zorder=5)  # Zorder for making the point appear on top of the line
    plt.annotate('Branch', xy=(0, 3), xytext=(0, -15), textcoords='offset points', ha='center', color='red')
    if output_filename is not None:
        plt.savefig(output_filename)
    if show_plot:
        plt.show()
    else:
        plt.clf()
