import matplotlib.pyplot as plt
import numpy as np


def plot_segments_and_annotate_angle_from_origin(angles):
    # Convert angles from degrees to radians and adjust from vertical downward
    angles_rad = [np.radians(270 - angle) for angle in angles]

    # Starting point
    x, y = 0, 0
    coords = [(x, y)]

    # Calculate coordinates of segment ends
    for angle in angles_rad:
        x += np.cos(angle)
        y += np.sin(angle)
        coords.append((x, y))

    # Unpack coordinates for plotting
    x_coords, y_coords = zip(*coords)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, marker='o')

    # Annotate angle from origin at each point
    for i in range(1, len(coords)):
        # Calculate the angle from the origin to the point
        angle_from_origin_rad = np.arctan2(y_coords[i], x_coords[i])
        # Convert to angle from vertical axis
        angle_from_origin_deg = np.degrees(angle_from_origin_rad) + 90

        plt.annotate(f"{angle_from_origin_deg:.2f}°", (x_coords[i], y_coords[i]))

    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True)
    plt.title('Plot of Segments with Angles from Origin')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.show()


# Example usage
angles = [-19.64533259171024, -14.235824697386347, -6.76369419197766, 17.698134125822524, -13.24460257389283,
          18.83611798693631, 4.3883020337202066, 12.718831056215551, -8.986773899051958, -6.608834252050816,
          1.9132409606008083, 4.061319912236776, 5.583761958370551, 22.29412324059759, 18.452554458754356,
          20.879627198898575, 34.92916064021324, 38.23709090433427, 53.722529227499926]
plot_segments_and_annotate_angle_from_origin(angles)
