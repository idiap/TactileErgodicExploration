"""
    Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
    Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

    This file is part of tactileErgodicExploration.

    tactileErgodicExploration is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    tactileErgodicExploration is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with tactileErgodicExploration. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
import open3d as o3d

def compute_coverage_residual(initial_heat_arr, coverage_arr):
    """
    Compute the normalized residual between the goal density and the
    coverage density.

    Args:
        initial_heat_arr (numpy.ndarray): Array of initial heat maps representing the goal density.
        coverage_arr (numpy.ndarray): Array of heat maps representing the coverage density.

    Returns:
        numpy.ndarray: Array of normalized residuals.
        numpy.ndarray: Array of goal densities.
    """
    goal_density = normalize_mat(initial_heat_arr)
    # [:,None] is required to match the dimensions for broadcasting
    residual_arr = goal_density[:, None] - coverage_arr
    uncovered_arr = np.where(residual_arr < 0, 0, residual_arr)
    # overcovered_arr = np.where(residual_arr > 0, 0, residual_arr)
    normalized_residual = np.linalg.norm(uncovered_arr, axis=0) / np.linalg.norm(
        goal_density
    )

    return normalized_residual
def normalize_mat(mat):
    """
    Normalize a matrix by dividing each element by the sum of all elements in the
      matrix.

    Parameters:
    mat (numpy.ndarray): The input matrix to be normalized.

    Returns:
    numpy.ndarray: The normalized matrix.

    """
    # avoid division by zero
    return mat / (np.sum(mat) + 1e-12)


def process_point_cloud(filename, param):
    class pcloud:
        pass  # c-style struct

    pcd_tmp = o3d.io.read_point_cloud(filename)
    pcd = pcd_tmp.voxel_down_sample(voxel_size=param.voxel_size)  # downsample
    pcloud.pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    pcloud.vertices = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    original_num_vertices = np.asarray(pcd_tmp.points).shape[0]

    print(
        f"Original Point cloud with {original_num_vertices}"
        + f" points is downsampled with voxel size {param.voxel_size}"
        + f"\n resulted in {len(pcd.points)} points"
    )
    # set the exploration target using the 'red' channel of the point cloud
    u0 = colors[:, 0]
    pcloud.u0 = np.where(u0 < 1, 0, 255)
    # compute the K-D tree for the nearest neighbor queries later
    pcloud.pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    pcloud.dt, pcloud.h = calculate_dt(pcloud.vertices, param.alpha)
    print(f"dt: {pcloud.dt:.3e}, h: {pcloud.h:.3e}, s: {param.voxel_size:.3e}")
    return pcloud


def calculate_edges(metric, num_neighbours=4):
    """
    Calculate the edges between points based on the KD-tree.

    Parameters:
    -----------
    metric: numpy.ndarray
        The metric to calculate the edges on.
    num_neighbours: int
        The number of nearest neighbors to consider.

    Returns:
    --------
    numpy.ndarray
        The edges between points.
    """
    # Calculate the KD-tree of the selected feature space
    tree = KDTree(metric)
    # Query the neighbourhoods for each point of the selected feature
    # space to each point
    d_kdtree, idx = tree.query(metric, k=num_neighbours)
    # Remove the first point in the neighborhood as this is just the
    # queried point itself
    idx = idx[:, 1:]

    # Create the edges array between all the points and their closest
    # neighbours
    point_numbers = np.arange(len(metric))
    # Repeat each point in the point numbers array the number of closest
    # neighbours -> 1,2,3,4... becomes 1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4...
    point_numbers = np.repeat(point_numbers, num_neighbours - 1)
    # Flatten  the neighbour indices array -> from [1,3,10,14], [4,7,17,23]
    # , ... becomes [1,3,10,4,7,17,23,...]
    idx_flatten = idx.flatten()
    # Create the edges array by combining the two other ones as a vertical
    # stack and transposing them to get the input that LineSet requires
    edges = np.vstack((point_numbers, idx_flatten)).T

    return edges


def calculate_dt(vertices, m=1):
    """
    Calculate the dt value using mean edge lengths.

    Parameters:
    -----------
    vertices: numpy.ndarray
        The vertices of the point cloud.
    m: float
        The scaling factor.

    Returns:
    --------
    float
        The calculated dt value.
    """
    edges = calculate_edges(vertices)
    edge_vectors = vertices[edges[:, 1], :] - vertices[edges[:, 0], :]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    h = np.mean(edge_lengths)
    dt = m * h**2
    return dt, h


def compute_tangent_space(neighbor_coords):
    """
    Compute the tangent space of a local neighborhood.

    Parameters:
    -----------
    neighbor_coords: numpy.ndarray
        The coordinates of the neighboring points.

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The normal vector, tangent vector 1, and tangent vector 2.
    """
    # Step 1: Fit a plane to the local neighborhood using least squares
    # plane equation ax + by + c = z, plane equation is similar to the line equation
    # y = ax+b this is why we don't have a coefficient for z in the plane equation
    # A = [x y 1]
    # x = [a b c]T
    # b = z
    A = np.column_stack(
        [
            neighbor_coords[:, 0],
            neighbor_coords[:, 1],
            np.ones_like(neighbor_coords[:, 0]),
        ]
    )
    b = neighbor_coords[:, 2]  # z coords.
    coefficients, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Step 2: Compute the normal vector of the fitted plane
    # this is the gradient of the plane equation

    normal_vector = np.array([coefficients[0], coefficients[1], -1.0])

    # Step 3: Choose two tangent vectors in the tangent plane
    # u-axis is perp to the normal -> dot product is 0
    tangent_vector1 = np.array([-coefficients[1], coefficients[0], 0])  # u-axis

    # second tangent vector is perp to the normal and to the first tangent vector
    tangent_vector2 = np.cross(normal_vector, tangent_vector1)  # v-axis

    # Step 4: Normalize the vectors
    tangent_vector1 /= np.linalg.norm(tangent_vector1)
    tangent_vector2 /= np.linalg.norm(tangent_vector2)
    normal_vector /= np.linalg.norm(normal_vector)

    return (
        coefficients,
        normal_vector,
        tangent_vector1,
        tangent_vector2,
    )


def project_points2tangent_space(
    agent_coords,
    neighbor_coords,
    coefficients,
    normal_vector,
    tangent_vector_1,
    tangent_vector_2,
):
    """
    Project the points onto the tangent space.

    Parameters:
    -----------
    agent_coords: numpy.ndarray
        The coordinates of the agent point.
    neighbor_coords: numpy.ndarray
        The coordinates of the neighboring points.
    normal_vector: numpy.ndarray
        The normal vector of the tangent plane.
    tangent_vector_1: numpy.ndarray
        The first tangent vector of the tangent plane.
    tangent_vector_2: numpy.ndarray
        The second tangent vector of the tangent plane.

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray]
        The projected points and the UV coordinates.
    """
    coords = np.zeros((neighbor_coords.shape[0] + 1, 3))
    coords[0, :] = agent_coords
    coords[1:, :] = neighbor_coords
    projected_points = np.zeros_like(coords)
    uv_coords = np.zeros((coords.shape[0], 2))
    # plane_point = np.average(coords, axis=0)
    # plane equation: ax + by + c = z
    # get a point on the plane, set x = y = 0 -> c = z
    plane_point = np.array([0, 0, coefficients[2]])

    projected_agent_positon = (
        agent_coords - np.dot(agent_coords - plane_point, normal_vector) * normal_vector
    )

    for i in range(1, coords.shape[0]):
        projected_points[i, :] = (
            coords[i, :]
            - np.dot(coords[i, :] - plane_point, normal_vector) * normal_vector
        )
        uv_coords[i, 0] = np.dot(
            projected_points[i, :] - projected_agent_positon,
            tangent_vector_1,
        )
        uv_coords[i, 1] = np.dot(
            projected_points[i, :] - projected_agent_positon,
            tangent_vector_2,
        )
    return (
        projected_agent_positon,
        projected_points[1:, :],
        uv_coords[1:, :],
    )


def fit_poly_surface(uv_coords, values, degree=3):
    """
    Fit a polynomial surface to the point cloud.

    Parameters:
    -----------
    uv_coords: numpy.ndarray
        The UV coordinates of the point cloud.
    values: numpy.ndarray
        The values of the point cloud.
    degree: int
        The degree of the polynomial.

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray]
        The coefficients of the polynomial and the transformed
        coordinates.
    """
    dists = np.linalg.norm(uv_coords, axis=1)
    eps = 1 / np.max(dists)
    weights = np.exp(-eps * dists**2)
    W = np.diag(weights)

    x = np.vstack(
        [
            uv_coords[:, 0],
            uv_coords[:, 1],
            np.ones_like(uv_coords[:, 0]),
        ]
    ).T
    poly = PolynomialFeatures(degree)
    X = poly.fit_transform(x)

    y = values
    coeffs = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return coeffs, X


def get_gradient_3rd_degree_polynomial(uv_coords, c, return_neighbors=False):
    """
    Compute the gradient of a scalar field defined on a point cloud using
    a 3rd degree polynomial.

    Process is based on the following paper:
    Crane, K., Weischedel, C., & Wardetzky, M. (2013). Geodesics in heat:
    A new approach to computing distance based on heat flow.
    ACM Transactions on Graphics, 32(5),
    152:1-152:11. https://doi.org/10.1145/2516971.2516977

    Parameters:
    -----------
    uv_coords: numpy.ndarray
        The UV coordinates of the point cloud.
    c: numpy.ndarray
        The coefficients of the polynomial.

    Returns:
    --------
    numpy.ndarray
        The gradient vectors at each vertex.
    """
    if return_neighbors:
        x0 = uv_coords[:, 0]
        x1 = uv_coords[:, 1]
        x2 = np.ones_like(uv_coords[:, 0])
    else:
        # If speed is an issue don't consider neighbors
        x0 = uv_coords[0, 0]
        x1 = uv_coords[0, 1]
        x2 = np.ones_like(uv_coords[0, 0])

    # Analytical gradients of the 3rd degree polynomial w.r.t. uv_coords
    grad3_u = (
        c[1]
        + 2 * c[4] * x0
        + c[5] * x1
        + c[6] * x2
        + 3 * c[10] * x0**2
        + 2 * c[11] * x0 * x1
        + 2 * c[12] * x0 * x2
        + c[13] * x1**2
        + c[14] * x1 * x2
        + c[15] * x2**2
    )
    grad3_v = (
        c[2]
        + c[5] * x0
        + 2 * c[7] * x1
        + c[8] * x2
        + 2 * c[11] * x0 * x1
        + 3 * c[16] * x1**2
        + 2 * c[17] * x1 * x2
        + c[18] * x2**2
    )

    grad_uv = np.array([grad3_u, grad3_v]).T

    return grad_uv


def get_gradient(
    agent_position,
    neighbor_coords,
    neighbor_ids,
    ut,
    return_neighbors=False,
):
    """
    Compute the gradient of a scalar field defined on a point cloud.

    Parameters:
    -----------
    agent_coords: numpy.ndarray
        The coordinates of the agent point.
    neighbor_ids: numpy.ndarray
        The indices of the neighboring points.
    neighbor_coords: numpy.ndarray
        The coordinates of the neighboring points.
    ut: numpy.ndarray
        The values of the scalar field.

    Returns:
    --------
    numpy.ndarray
        The gradient vectors at each vertex.
    """
    (
        coefficients,
        normal_vector,
        tangent_vector_1,
        tangent_vector_2,
    ) = compute_tangent_space(neighbor_coords)

    (
        projected_agent_positon,
        projected_neighbor_coords,
        uv_coords,
    ) = project_points2tangent_space(
        agent_position,
        neighbor_coords,
        coefficients,
        normal_vector,
        tangent_vector_1,
        tangent_vector_2,
    )

    values = np.zeros_like(neighbor_ids, dtype=float)

    for i in range(len(neighbor_ids)):
        values[i] = ut[neighbor_ids[i]]

    # consider the temperature values on the tangent
    # space as heights and fit a 3rd degree polynomial
    coeffs, X = fit_poly_surface(uv_coords, values)
    # get gradient in the tangent space (uv-coords)
    grad_uv = get_gradient_3rd_degree_polynomial(uv_coords, coeffs, return_neighbors)

    # project gradient back to 3-D
    if return_neighbors:  # if speed is an issue don't consider neighbors
        grad_3d = (
            grad_uv[:, 0][:, None] * tangent_vector_1
            + grad_uv[:, 1][:, None] * tangent_vector_2
        )
        unit_grad = grad_3d / np.linalg.norm(grad_3d, axis=1)[:, None]
    else:
        grad_3d = grad_uv[0] * tangent_vector_1 + grad_uv[1] * tangent_vector_2
        unit_grad = grad_3d / np.linalg.norm(grad_3d)

    return (
        projected_agent_positon,
        unit_grad,
        projected_neighbor_coords,
    )


def get_pcloud_neighbors(
    pcd_tree,
    vertices,
    agent_position,
    agent_radius,
    neighbor_size_limit,
    nb_minimum_neighbors,
):
    """
    Compute the kNN neighbors of the agent from the point cloud based on the
    Euclidean distance.

    Parameters:
    - pcd_tree: The point cloud KDTree object.
    - vertices: The vertices of the point cloud.
    - agent_position: The position of the agent.
    - agent_radius: The radius of the agent.
    - neighbor_size_limit: The maximum size of the neighborhood.
    - nb_minimum_neighbors: The minimum number of neighbors required.

    Returns:
    - dists: The distances between the agent and its neighbors.
    - neighbor_ids: The indices of the neighboring points.
    - neighbor_coords: The coordinates of the neighboring points.
    """
    # this returns all the neighbors within the radius
    # but it is limited by the neighbor_size_limit
    [k, idx, dists] = pcd_tree.search_hybrid_vector_3d(
        agent_position, agent_radius, neighbor_size_limit
    )

    # if we don't have enough neighbors, we search for the k nearest neighbors
    if k <= nb_minimum_neighbors:
        [k, idx, dists] = pcd_tree.search_knn_vector_3d(
            agent_position, nb_minimum_neighbors
        )

    neighbor_ids = np.asarray(idx)
    dists = np.asarray(dists)
    neighbor_coords = vertices[neighbor_ids, :]
    return dists, neighbor_ids, neighbor_coords
