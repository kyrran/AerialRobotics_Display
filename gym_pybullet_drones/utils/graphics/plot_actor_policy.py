import matplotlib.pyplot as plt
import numpy as np


def visualize_policy(model, buffer=None, action_scale=1.0):
    # Grid of states
    x_vals = np.linspace(-3, 3, 20)
    z_vals = np.linspace(0, 6, 20)
    X, Z = np.meshgrid(x_vals, z_vals)

    # Policy outputs
    U = np.zeros_like(X)
    V = np.zeros_like(Z)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = np.array([X[i, j], Z[i, j]])

            action, _ = model.predict(state)

            print("Action:", action)
            U[i, j] = action[0] * action_scale
            V[i, j] = action[1] * action_scale

    if buffer is not None:
        # Separate coordinates and actions from the buffer
        coords = np.array([pos for pos, _, _, _, _, _ in buffer])
        actions = np.array([rew for _, _, rew, _, _, _ in buffer])

        X_buf, Z_buf = coords[:, 0], coords[:, 1]
        U_buf, V_buf = actions[:, 0], actions[:, 1]

    plt.figure(figsize=(12, 10))

    if buffer is not None:
        plt.quiver(X_buf, Z_buf, U_buf, V_buf, angles="xy", color='green', label="Buffer Actions")

    # Plot the model's policy as a quiver plot
    plt.quiver(X, Z, U, V, angles="xy", color='red', label="SAC Policy")

    plt.title("Policy Visualization")
    plt.xlabel("X axis")
    plt.ylabel("Z axis")
    plt.xlim([-3.2, 3.2])
    plt.ylim([-0.2, 6.2])
    plt.grid(True)

    plt.show()
