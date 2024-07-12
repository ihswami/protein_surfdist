from typing import Tuple
from typing import NoneType
from numpy.typing import NDArray
import numpy as np
import igl
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
import potpourri3d as pp3d
import trimesh




def compute_mesh_LM(vertices: NDArray, faces: NDArray) -> Tuple[NDArray, NDArray]:
    """Compute mesh cotangent and mass matrices."""
    L = igl.cotmatrix(vertices, faces)
    M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    return L, M


def compute_LM_from_point_cloud(vertices: NDArray, faces: NoneType=None) -> Tuple[NDArray, NDArray]:
    """Compute cotangent and mass matrices from point cloud."""
    tri = Delaunay(vertices)
    vertices, faces = vertices, tri.simplices    
    L, M = igl.cotmatrix(vertices, faces), igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)
    return L, M


def get_evec_eval_using_numpy_method(L: csr_matrix, M: csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes sparse L and M and returns eigenvalues and eigenvectors sorted by largest magnitude of eigenvalues.

    Parameters:
    L (csr_matrix): A sparse matrix.
    M (csr_matrix): A sparse matrix to be used for the inverse.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the eigenvalues and eigenvectors.
    """
    L_dense = L.toarray()
    M_dense_inv = np.sqrt(np.linalg.inv(M.toarray()))
    eigenvalues, eigenvectors = np.linalg.eig((M_dense_inv @ L_dense) @ M_dense_inv) 
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]  # Get indices that sort the magnitude of eigenvalues in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = np.linalg.inv(M_dense_inv) @ eigenvectors[:, sorted_indices]
    return eigenvalues, eigenvectors


def mean_curvature_flow(mesh, num_iterations: int=10, time_step: float=0.01) -> trimesh:
    num_vertices = len(mesh.vertices)
    laplacian_matrix = trimesh.smoothing.laplacian_calculation(mesh)
    vertices = mesh.vertices.copy()

    for _ in range(num_iterations):
        mean_curvature_normal = laplacian_matrix @ vertices
        vertices += time_step * mean_curvature_normal
        # # Recompute the vertex positions
        # vertices = spsolve(csr_matrix(np.eye(num_vertices) - time_step * laplacian_matrix), vertices)
    mesh.vertices = vertices
    return mesh


def get_smooth_mesh(surface_file: str, num_iterations: int=100, time_step: float=0.01) -> Tuple[NDArray, NDArray]:
    """
    Smooths a 3D mesh using Mean Curvature Flow.

    Parameters:
    - Surface file
    - num_iterations: int, optional (default=100)
        The number of iterations to run the smoothing algorithm.
    - time_step: float, optional (default=0.01)
        The time step for each iteration of the smoothing algorithm.

    Returns:
    - smoothed_vertices: ndarray of shape (n, 3)
        The smoothed vertex coordinates of the mesh.
    - smoothed_faces: ndarray of shape (m, 3)
        The indices of the vertices forming the triangular faces of the smoothed mesh.
    """
    # Create a trimesh object from the given vertices and faces
    vertices, faces = pp3d.read(surface_file)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Apply mean curvature flow to the mesh
    smoothed_mesh = mean_curvature_flow(mesh, num_iterations=num_iterations, time_step=time_step)
    
    # Unpack vertices and faces from the smoothed mesh
    smoothed_vertices = smoothed_mesh.vertices
    smoothed_faces = smoothed_mesh.faces

    # Return the smoothed vertices and faces
    return smoothed_vertices, smoothed_faces
