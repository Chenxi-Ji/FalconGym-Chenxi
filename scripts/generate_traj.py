import os
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import yaml
import json
from scipy.spatial.transform import Rotation

from utils_rotation import orientation_to_direction, direction_to_orientation


### Generate the smooth trajectory using TCB splines
def compute_tcb_tangents(points, directions, tension=0.0, continuity=0.0, bias=0.0):
    """
    Compute Kochanek–Bartels tangents for all points.
    """
    n = len(points)
    tangents = np.zeros_like(points)

    # Arc-length parameterization
    dist = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(dist)])
    t /= t[-1]

    for i in range(n):
        if i == 0:
            tangents[i] = directions[i]
            continue
        if i == n - 1:
            tangents[i] = directions[i]
            continue

        dt_prev = t[i] - t[i - 1]
        dt_next = t[i + 1] - t[i]

        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]

        d_prev = (p_curr - p_prev) / dt_prev
        d_next = (p_next - p_curr) / dt_next

        t_in = (
            (1 - tension) * (1 + bias) * (1 + continuity) / 2 * d_prev +
            (1 - tension) * (1 - bias) * (1 - continuity) / 2 * d_next
        )

        t_out = (
            (1 - tension) * (1 - bias) * (1 + continuity) / 2 * d_prev +
            (1 - tension) * (1 + bias) * (1 - continuity) / 2 * d_next
        )

        # Blend geometric tangent with desired direction
        mag = 0.5 * (np.linalg.norm(d_prev) + np.linalg.norm(d_next))
        dunit = directions[i] / (np.linalg.norm(directions[i]) + 1e-6)

        tangents[i] = 0.7 * t_out + 0.3 * dunit * mag

    return tangents, t


def hermite_eval(p0, p1, m0, m1, t):
    """Evaluate cubic Hermite spline"""
    t2 = t * t
    t3 = t2 * t

    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


### Generate the smooth trajectory
def generate_smooth_trajectory(
    start_pose,
    end_pose,
    gate_poses,
    num_waypoints=500,
    tension=0.2,
    continuity=0.0,
    bias=0.0,
    
):
    """
    TCB spline trajectory through gates with smooth curvature.

    Returns:
        np.ndarray: The trajectory points.
        np.ndarray: The tangents (directions) at each trajectory point.
    """
    points = np.array(
        [start_pose[:3]] +
        [g[:3] for g in gate_poses] +
        [end_pose[:3]]
    )

    orientations = np.array(
        [start_pose[3:]] +
        [g[3:] for g in gate_poses] +
        [end_pose[3:]]
    )

    # Compute direction vectors from orientations
    directions = np.array([
        orientation_to_direction(orientations[i, 0],
                                orientations[i, 1],
                                orientations[i, 2])
        for i in range(len(orientations))
    ])

    tangents, t = compute_tcb_tangents(
        points, directions, tension, continuity, bias
    )

    # Sample trajectory
    ts_dense = np.linspace(0, 1, num_waypoints)
    traj = []
    traj_tangents = []

    for s in ts_dense:
        i = np.searchsorted(t, s) - 1
        i = np.clip(i, 0, len(points) - 2)

        local_t = (s - t[i]) / (t[i + 1] - t[i] + 1e-8)

        p = hermite_eval(
            points[i],
            points[i + 1],
            tangents[i] * (t[i + 1] - t[i]),
            tangents[i + 1] * (t[i + 1] - t[i]),
            local_t
        )

        tangent = hermite_eval(
            tangents[i],
            tangents[i + 1],
            np.zeros_like(tangents[i]),
            np.zeros_like(tangents[i + 1]),
            local_t
        )

        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

        traj.append(p)
        traj_tangents.append(tangent)

    return np.array(traj), np.array(traj_tangents), 

def generate_dynamic_radius_profile(trajectory, gate_poses, base_radius, expand_radius):
    # Compute dynamic radius along the trajectory
    dynamic_radius_set = []
    gate_indices = [0] + [np.argmin(np.linalg.norm(trajectory - np.array(g[:3]), axis=1)) for g in gate_poses] + [len(trajectory) - 1]
 
    # Plot the tube around the trajectory using circles orthogonal to the tangents
    for i, (point, tangent) in enumerate(zip(trajectory, tangents)):
        # Determine the current segment (start to first gate, between gates, or last gate to end)
        if i <= gate_indices[1]:  # Between start_pose and the first gate
            dynamic_radius = base_radius
        elif i >= gate_indices[-2]:  # Between the last gate and end_pose
            dynamic_radius = base_radius
        else:  # Between two gates
            prev_gate_pose = trajectory[max([gate_indice for gate_indice in gate_indices if gate_indice <= i], default=None)]
            next_gate_pose = trajectory[min([gate_indice for gate_indice in gate_indices if gate_indice > i], default=None)]

            point = trajectory[i]
            d = np.linalg.norm(next_gate_pose - prev_gate_pose)
            xl = np.linalg.norm(point - prev_gate_pose)
            xn = np.linalg.norm(point - next_gate_pose)
            dynamic_radius = base_radius + (xl * xn) / (d * d) * expand_radius * 4
        dynamic_radius_set.append(dynamic_radius)

    return np.array(dynamic_radius_set)

### Generate sample points around the trajectory
def generate_samples(trajectory, tangents, dynamic_radius_set, N_samples, direction):
    """
    Generate sample points around the trajectory within a cylindrical range.

    Args:
        trajectory (np.ndarray): The smooth trajectory as an array of waypoints.
        tangents (np.ndarray): The tangents (directions) at each trajectory point.
        dynamic_radius_set (np.ndarray): The dynamic radius at each trajectory point.
        N_samples (int): Number of samples to generate per trajectory point.
        direction (np.ndarray): Direction vector for the cylinder height at each trajectory point.

    Returns:
        np.ndarray: Array of sampled points.
    """
    samples = []

    for i, (point, tangent, dynamic_radius) in enumerate(zip(trajectory, tangents, dynamic_radius_set)):
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)  # Normalize the tangent

        # Find two orthogonal vectors to the tangent
        if np.allclose(tangent, [1, 0, 0]) or np.allclose(tangent, [-1, 0, 0]):
            orthogonal1 = np.array([0, 1, 0])
        else:
            orthogonal1 = np.cross(tangent, [1, 0, 0])
            orthogonal1 = orthogonal1 / (np.linalg.norm(orthogonal1) + 1e-8)

        orthogonal2 = np.cross(tangent, orthogonal1)

        # Generate samples within the cylinder
        direction_vector = direction[i]  # Use the provided direction for the cylinder
        direction_coeff = np.linspace(0, 1, N_samples)  # Distribute samples along the cylinder height

        for coeff in direction_coeff:
            # Generate a random point within the circle at this height
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, dynamic_radius)  # Ensure samples are within the circle
            sampled_point = (
                point +
                coeff * direction_vector +  # Move along the cylinder height
                r * np.cos(theta) * orthogonal1 +
                r * np.sin(theta) * orthogonal2
            )

            samples.append(sampled_point)

    return np.array(samples)

### Plot
def generate_gate_circle(point, tangent, outer_radius=0.2, height=0.1, num_points=100, num_slices=5):
    """
    Generate a hollow cylindrical region representing a gate with multiple circles between the top and bottom surfaces.

    Args:
        point (list): Center of the gate [x, y, z].
        tangent (list): Tangent vector [dx, dy, dz] defining the gate's orientation.
        outer_radius (float): Outer radius of the gate.
        height (float): Height of the cylindrical region.
        num_points (int): Number of points to generate the circle.
        num_slices (int): Number of slices between the top and bottom surfaces.

    Returns:
        np.ndarray: Points representing the hollow cylindrical region.
    """
    inner_radius = outer_radius * 0.85  # Define inner radius for hollow effect
    # Normalize tangent
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

    # Find two orthogonal vectors to the tangent
    if np.allclose(tangent, [1, 0, 0]) or np.allclose(tangent, [-1, 0, 0]):
        orthogonal1 = np.array([0, 1, 0])
    else:
        orthogonal1 = np.cross(tangent, [1, 0, 0])
        orthogonal1 = orthogonal1 / np.linalg.norm(orthogonal1)

    orthogonal2 = np.cross(tangent, orthogonal1)

    # Generate points for the outer and inner circles
    theta = np.linspace(0, 2 * np.pi, num_points)
    outer_circle = (
        outer_radius * np.outer(np.cos(theta), orthogonal1) +
        outer_radius * np.outer(np.sin(theta), orthogonal2)
    )
    inner_circle = (
        inner_radius * np.outer(np.cos(theta), orthogonal1) +
        inner_radius * np.outer(np.sin(theta), orthogonal2)
    )

    # Generate slices between the top and bottom surfaces
    slice_positions = np.linspace(-height / 2, height / 2, num_slices)
    hollow_cylinder = []

    for z in slice_positions:
        hollow_cylinder.append(outer_circle + np.array(point) + z * tangent)
        hollow_cylinder.append(inner_circle + np.array(point) + z * tangent)

    # Combine all slices into a single array
    hollow_cylinder = np.vstack(hollow_cylinder)

    return hollow_cylinder

def generate_cylinder(point, tangent, direction, radius=0.2, num_points=100, num_slices=10):
    """
    Generate a 3D cylinder representing a gate based on its tangent and direction.

    Args:
        point (list): Center of the gate [x, y, z].
        tangent (list): Tangent vector [dx, dy, dz] defining the gate's orientation.
        direction (list): Direction vector [dx, dy, dz] defining the cylinder's height.
        radius (float): Radius of the cylinder.
        num_points (int): Number of points to generate the circle.
        num_slices (int): Number of slices along the direction to mimic the cylinder.

    Returns:
        np.ndarray: Points representing the cylinder.
    """
    # Normalize tangent
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

    # Find two orthogonal vectors to the tangent
    if np.allclose(tangent, [1, 0, 0]) or np.allclose(tangent, [-1, 0, 0]):
        orthogonal1 = np.array([0, 1, 0])
    else:
        orthogonal1 = np.cross(tangent, [1, 0, 0])
        orthogonal1 = orthogonal1 / np.linalg.norm(orthogonal1)

    orthogonal2 = np.cross(tangent, orthogonal1)

    # Generate points for the base circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    base_circle = (
        radius * np.outer(np.cos(theta), orthogonal1) +
        radius * np.outer(np.sin(theta), orthogonal2)
    )

    # Generate slices along the direction
    direction_coeff = np.linspace(0, 1, num_slices)
    cylinder_points = []
    for coeff in direction_coeff:
        slice_circle = base_circle + np.array(point) + coeff * np.array(direction)
        cylinder_points.append(slice_circle)

    # Combine all slices into a single array
    cylinder_points = np.vstack(cylinder_points)

    return cylinder_points

def plot_trajectory(trajectory, tangents, start_pose, end_pose, gate_poses, gate_radius, samples=None, dynamic_radius_set=None):
    """
    Plot the smooth trajectory, a tube around it, gate positions as circles, start point, end point, and optionally sample points.

    Args:
        trajectory (np.ndarray): The smooth trajectory as an array of waypoints.
        tangents (np.ndarray): The tangents (directions) at each trajectory point.
        gate_poses (list): A list of gate positions, each as [x, y, z, dx, dy, dz].
        start_pose (list): The starting point of the trajectory [x, y, z, dx, dy, dz].
        end_pose (list): The ending point of the trajectory [x, y, z, dx, dy, dz].
        gate_radius (float): Radius of the gates.
        base_radius (float): Base radius of the tube around the trajectory.
        expand_radius (float): Rate to adjust the tube radius based on the distance traveled.
        save_samples (bool): Whether to plot sample points.
        samples_file (str): Path to the samples JSON file.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Traj', color='black')

    # Plot the gates as circles based on their tangents
    for i, gate in enumerate(gate_poses):
        gate_point = gate[:3]
        gate_tangent = orientation_to_direction(*gate[3:])
        circle = generate_gate_circle(gate_point, gate_tangent, gate_radius)
        if i == 0:  # Add label only for the first gate
            ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color='blue', label='Gate')
        else:
            ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color='blue')

    # Initialize variables for dynamic radius calculation
    if samples is not None: 
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='orange', label='Samples', alpha=0.5, s=10)

    if dynamic_radius_set is not None:
        for i, (point, tangent, dynamic_radius) in enumerate(zip(trajectory, tangents, dynamic_radius_set)):
            direction = trajectory[min(i + 1, len(trajectory) - 1)] - point  # Direction for the cylinder
            cylinder = generate_cylinder(point, tangent, direction, dynamic_radius)
            if i == 0:  # Add label only for the first tube
                ax.plot(cylinder[:, 0], cylinder[:, 1], cylinder[:, 2], color='cyan', alpha=0.3, label='ODD')
            else:
                ax.plot(cylinder[:, 0], cylinder[:, 1], cylinder[:, 2], color='cyan', alpha=0.3)

    # Plot the start point
    ax.scatter(*start_pose[:3], color='green', label='Start Point', s=10)

    # Plot the end point
    ax.scatter(*end_pose[:3], color='purple', label='End Point')
        

    # # Set axis limits to the same range
    # x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
    # y_min, y_max = np.min(trajectory[:, 1]), np.max(trajectory[:, 1])
    # z_min, z_max = np.min(trajectory[:, 2])
    # max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

    # mid_x = (x_max + x_min) / 2.0
    # mid_y = (y_max + y_min) / 2.0
    # mid_z = (z_max + z_min) / 2.0

    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # Compress the z-axis to 1/3 of its original range
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    ax.set_box_aspect([1, 1, (z_limits[1] - z_limits[0]) / 3])

    # ax.legend()
    plt.show()

def save_trajectory(trajectory, tangents, gate_poses, dynamic_radius_set, N_samples, samples, traj_file="traj.json", samples_file="samples.json"):
    """
    Save the trajectory as a dictionary with fields (x, y, z, dx, dy, dz, gate_x, gate_y, gate_z, index, radius).
    If save_samples is True, also save sampled points around each trajectory point.

    Args:
        trajectory (np.ndarray): The smooth trajectory as an array of waypoints.
        tangents (np.ndarray): The tangents (directions) at each trajectory point.
        gate_poses (list): A list of gate positions, each as [x, y, z, dx, dy, dz].
        output_file (str): The file to save the trajectory.
        base_radius (float): Base radius of the tube.
        expand_radius (float): Rate to adjust the tube radius based on the distance traveled.
        save_samples (bool): Whether to save sampled points around each trajectory point.
        N_samples (int): Number of samples to generate around each trajectory point.
    """
    data = []
    samples_data = []
    num_gates = len(gate_poses)

    # Precompute gate indices
    gate_indices = [0] + [np.argmin(np.linalg.norm(trajectory - np.array(g[:3]), axis=1)) for g in gate_poses] + [len(trajectory) - 1]
    for i, (point, tangent, dynamic_radius) in enumerate(zip(trajectory, tangents,dynamic_radius_set)):

        if i < gate_indices[1]:  # Between start_pose and the first gate
            gate_pose = gate_poses[0]
        elif i >= gate_indices[-2]:  # Between the last gate and end_pose
            gate_pose = None
        else:  # Between two gates
            indice = next((ind for ind, g in enumerate(gate_indices) if g > i),None)
            gate_pose = gate_poses[indice-1]#trajectory[indice]

        # orientation = direction_to_orientation(tangent[0], tangent[1], tangent[2])

        diff_point = np.array(gate_pose[:3]) - np.array(point) if gate_pose is not None else np.array(end_pose[:3]) - np.array(point)
        orientation = direction_to_orientation(*diff_point) 

        rot = Rotation.from_euler("ZYX", orientation)
        R_matrix = rot.as_matrix()
        # Save trajectory point data
        data.append({
            "index": i,
            "pose": [*point, *orientation],
            "tangent": [*tangent],
            "gate": [*gate_pose] if gate_pose is not None else None,
            "radius": dynamic_radius
        })

        if samples is not None:
            sample_index_start = i * N_samples

            for j in range(N_samples):
                sampled_point = samples[sample_index_start + j]

                diff_point = np.array(gate_pose[:3]) - np.array(sampled_point) if gate_pose is not None else np.array(end_pose[:3]) - np.array(sampled_point)
                relative_pose = R_matrix @ diff_point

                samples_data.append({
                    "index": sample_index_start + j,
                    "pose": [*sampled_point, *orientation],
                    "gate": [*gate_pose] if gate_pose is not None else None,
                    "relative_pose": [*relative_pose]

                })

    # Save the trajectory data as a JSON file
    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    with open(traj_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Trajectory saved to {traj_file}")

    # Save the sampled points data as a JSON file if save_samples is True
    if samples is not None:
        os.makedirs(os.path.dirname(traj_file), exist_ok=True)
        with open(samples_file, "w") as f:
            json.dump(samples_data, f, indent=4)
        print(f"Sampled points saved to {samples_file}")

if __name__ == "__main__":
    import argparse
    ### Default command: python3 scripts/generate_traj.py --config configs/${case_name}/traj.yaml
    parser = argparse.ArgumentParser(description="Abstract Gsplat with YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load parameters from YAML file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)


    start_pose = config["start_pose"]
    gate_poses = config["gate_poses"]  # Use full 6D gate positions
    end_pose = config["end_pose"]
    num_waypoints = config["num_waypoints"]  # Default to 200 waypoints if not specified
    gate_radius = config["gate_radius"]  # Default gate radius
    base_radius = config["base_radius"]  # Load radius from the configuration
    expand_radius = config["expand_radius"]  # Load expand_radius from the configuration
    save_samples = config["save_samples"]
    N_samples = config["N_samples"]
    case_name = config["case_name"]

    # Generate the smooth trajectory
    trajectory, tangents = generate_smooth_trajectory(start_pose, end_pose, gate_poses, num_waypoints)

    dynamic_radius_set = generate_dynamic_radius_profile(trajectory, gate_poses, base_radius, expand_radius)

    # Compute direction vectors for the cylinder sampling
    direction = np.array([
        trajectory[min(i + 1, len(trajectory) - 1)] - trajectory[i]
        for i in range(len(trajectory))
    ])

    samples = generate_samples(trajectory, tangents, dynamic_radius_set, N_samples, direction) if save_samples else None

    # Plot the trajectory with a tube and optionally sample points
    plot_trajectory(trajectory, tangents, start_pose, end_pose, gate_poses, gate_radius, samples, dynamic_radius_set)


    traj_file = f"results/{case_name}/traj.json"
    samples_file = f"results/{case_name}/samples.json" if save_samples else None
    
    # Save the trajectory to a file
    save_trajectory(trajectory, tangents, gate_poses, dynamic_radius_set, N_samples, samples, traj_file, samples_file)
