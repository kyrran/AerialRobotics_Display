import numpy as np
import pandas as pd
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

def load_waypoints(filename):
    """Load waypoints from a CSV file."""
    try:
        data = pd.read_csv(filename)
        # data[['x', 'y', 'z']] -= [0,0,2.7]
        # data[['x', 'y', 'z']] += [1.6,0,1.8]

        waypoints = data[['x', 'y', 'z']].values  # Extract x, y, z columns
        # data['drone_x'] = data['drone_x'] / 1000.00
        # data['drone_y'] = data['drone_y'] / 1000.00
        # data['drone_z'] = data['drone_z'] / 1000.00
        
        # waypoints = data[['drone_x', 'drone_y', 'drone_z']].values
        
        # print(waypoints)
        return waypoints
    except Exception as e:
        print(f"Error loading waypoints: {e}")
        return None

def process_trajectory(file_path, processed_file_path, initial_hover_point):
    
    # Extract x, y, z columns for processing
    waypoints = load_waypoints(file_path)
    
    initial_hover_point = np.array([initial_hover_point])
    waypoints = np.vstack((initial_hover_point, waypoints))

    print("Number of waypoints:", len(waypoints))

    # Define the velocity and acceleration limits
    vlim = np.array([1, 1, 1])  # velocity limits in each axis
    alim = np.array([0.5, 0.5, 0.5])  # acceleration limits in each axis

    # Create path from waypoints
    path = ta.SplineInterpolator(np.linspace(0, 1, len(waypoints)), waypoints)
    
    # Check the created path
    print("Created Path:", path)

    # Create velocity and acceleration constraints
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)

    # Setup the parameterization problem
    instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')

    # Compute the trajectory
    try:
        jnt_traj = instance.compute_trajectory(0, 0)
        if jnt_traj is None or jnt_traj.duration is None:
            raise ValueError("Trajectory computation failed, resulting in an invalid trajectory.")
        print("Trajectory duration:", jnt_traj.duration)
    except Exception as e:
        print(f"Error during trajectory computation: {e}")
        raise

    # Sample the trajectory
    N_samples = 150
    ss = np.linspace(0, jnt_traj.duration, N_samples)
    qs = jnt_traj(ss)

    # Extract the x, y, z components of the trajectory
    x = qs[:, 0]
    y = qs[:, 1]
    z = qs[:, 2]

    # Save the processed waypoints to a new CSV file
    processed_waypoints = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z
    })
    processed_waypoints.to_csv(processed_file_path, index=False)

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def process_trajectory_with_cubic_spline(file_path, processed_file_path, initial_hover_point):
    # Load the waypoints from the file
    waypoints = load_waypoints(file_path)
    
    # Append the initial hover point to the waypoints
    initial_hover_point = np.array([initial_hover_point])
    waypoints = np.vstack((initial_hover_point, waypoints))
    
    # Extract x, y, z components
    x_points = waypoints[:, 0]
    y_points = waypoints[:, 1]
    z_points = waypoints[:, 2]
    
    # Create an array of parameter values (usually time or distance) for the points
    t = np.linspace(0, 1, len(waypoints))
    
    # Create cubic splines for x, y, z coordinates
    spline_x = CubicSpline(t, x_points)
    spline_y = CubicSpline(t, y_points)
    spline_z = CubicSpline(t, z_points)
    
    # Generate a smooth trajectory with a higher resolution
    t_new = np.linspace(0, 1, 700)  # Increase the number of points for smoothness
    x_new = spline_x(t_new)
    y_new = spline_y(t_new)
    z_new = spline_z(t_new)
    
    # Save the processed waypoints to a new CSV file
    processed_waypoints = pd.DataFrame({
        'x': x_new,
        'y': y_new,
        'z': z_new
    })
    processed_waypoints.to_csv(processed_file_path, index=False)
    
    # Optional: Plot the original and interpolated points for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, z_points, 'o', label='Original waypoints')
    plt.plot(x_new, z_new, '-', label='Cubic Spline Interpolated Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Spline Interpolation of Waypoints')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    
def toppra_waypoints(simplified_waypoints, num_samples):
    
    """Toppra Algorithm to smooth the trajectory"""
    
    # Extract x, y, z columns for processing
    waypoints = simplified_waypoints
    
    # Define the velocity and acceleration limits
    vlim = np.array([1, 1, 1])  # velocity limits in each axis
    alim = np.array([0.5, 0.5, 0.5])  # acceleration limits in each axis

    # Create path from waypoints
    path = ta.SplineInterpolator(np.linspace(0, 1, len(waypoints)), waypoints)

    # Create velocity and acceleration constraints
    pc_vel = constraint.JointVelocityConstraint(vlim)
    pc_acc = constraint.JointAccelerationConstraint(alim)

    # Setup the parameterization problem
    instance = algo.TOPPRA([pc_vel, pc_acc], path, solver_wrapper='seidel')

    # Compute the trajectory
    jnt_traj = instance.compute_trajectory(0, 0)

    # Sample the trajectory
    N_samples = num_samples
    ss = np.linspace(0, jnt_traj.duration, N_samples)
    qs = jnt_traj(ss)

    # Extract the x, y, z components of the trajectory
    x = qs[:, 0]
    y = qs[:, 1]
    z = qs[:, 2]

    # Save the processed waypoints to a new CSV file
    processed_waypoints = pd.DataFrame({
        'drone_x': x,
        'drone_y': y,
        'drone_z': z
    })
    
    return processed_waypoints

