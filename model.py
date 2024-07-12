import os
import time
import pickle
from dataclasses import dataclass
from typing import List
from scipy.sparse.linalg import eigsh
import torch
import numpy as np
import pyFM.signatures as sg

from utils import geometry as geo
from utils import misc
from utils import metrics

def cache_results(func): 
    def wrapper(self, *args, **kwargs):
        path_save = self._get_path_save(*args, **kwargs)
        if os.path.exists(path_save):
            return pickle.load(open(path_save, 'rb'))
        result = func(self, *args, **kwargs)
        pickle.dump(result, open(path_save, 'wb'))
        return result
    return wrapper

def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")
        return result
    return wrapper

@dataclass
class SurfaceDistanceUsingFmap:
    cache_path: str = './cache'
    path_save: str = './evals'
    data_path: str = './data'

    @time_execution
    @cache_results
    def compute_Fmaps_n_distance(self, surface1: str, surface2: str, point_cloud: bool = False,
                                 k: int = 500, smooth_iteration: int = 10, which: str = 'LM', 
                                 signature: str = "HKS", time_intv: List[float] = [0.0001, 0.01, 0.1, 1], 
                                 num_E: int = 3) -> tuple[any]:
        
        surface1_path = os.path.join(self.data_path, surface1)
        surface2_path = os.path.join(self.data_path, surface2)

        verts1, faces1, verts2, faces2 = None, None, None, None
        if not point_cloud:
            verts1, faces1 = geo.get_smooth_mesh(surface1_path, num_iterations=smooth_iteration, time_step=0.01)
            verts2, faces2 = geo.get_smooth_mesh(surface2_path, num_iterations=smooth_iteration, time_step=0.01)

        if point_cloud:
            # create this function.
            # Also first one has to create a surface class which preprocesses the surface data and converts it into a predecided standard form for the data.  
            verts1 = geo.read_point_cloud(surface1_path)
            verts2 = geo.read_point_cloud(surface2_path)


        eigenvalues1, eigenvectors1, eigenvalues2, eigenvectors2 = self._compute_eigen(surface1, verts1, faces1, surface2, verts2, faces2, k, which, point_cloud)

        signature1, signature2 = self._compute_signatures(signature, eigenvalues1, eigenvectors1, eigenvalues2, eigenvectors2, k, time_intv, num_E)
        
        A = torch.tensor(eigenvectors1[:, :k]).T @ signature1
        B = torch.tensor(eigenvectors2[:, :k]).T @ signature2
        
        best_C, best_D, min_loss = misc.perform_optimization(A, B, eigenvalues1, eigenvalues2, k)
        
        # the following is a task dependent function and create function for it for different problem dataset.
        dist_F_JS = metrics.compute_all_distances(best_C, best_D, k, surface1, surface2, eigenvectors1, eigenvectors2)
        map_metric = metrics.compute_mapping_metric(best_C, best_D, eigenvalues1, eigenvalues2, k)
        
        return best_C.detach().numpy(), best_D.detach().numpy(), dist_F_JS, min_loss, map_metric

    def _get_path_save(self, surface1, surface2, k, **kwargs):
        return f"{self.path_save}/{surface1}_{surface2}_{k}.pkl"

    def _compute_eigen(self, surface1, verts1, faces1, surface2, verts2, faces2, k, which, point_cloud):
        cache_path1 = f'{self.cache_path}/scipyevecval/{surface1}_{which}_{k}.pkl'
        cache_path2 = f'{self.cache_path}/scipyevecval/{surface2}_{which}_{k}.pkl'
        
        def load_or_compute_eigen(cache_path, verts, faces, compute_fn):
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as file:
                    return pickle.load(file)
            L, M = compute_fn(verts, faces)
            eigenvalues, eigenvectors = eigsh(L, k=k, M=M, which=which)
            with open(cache_path, 'wb') as file:
                pickle.dump((eigenvalues, eigenvectors), file)
            return eigenvalues, eigenvectors

        compute_fn = geo.compute_LM_from_point_cloud if point_cloud else geo.compute_mesh_LM

        if surface1 == surface2 and os.path.exists(cache_path1):
            eigenvalues1, eigenvectors1 = pickle.load(open(cache_path1, 'rb'))
            return eigenvalues1, eigenvectors1, eigenvalues1, eigenvectors1

        eigenvalues1, eigenvectors1 = load_or_compute_eigen(cache_path1, verts1, faces1, compute_fn)
        eigenvalues2, eigenvectors2 = load_or_compute_eigen(cache_path2, verts2, faces2, compute_fn)
        
        return eigenvalues1, eigenvectors1, eigenvalues2, eigenvectors2

    def _compute_signatures(self, signature, eigenvalues1, eigenvectors1, eigenvalues2, eigenvectors2, k, time_intv, num_E):
        if signature == "WKS":
            return (sg.auto_WKS(torch.tensor(eigenvalues1[:k]), torch.tensor(eigenvectors1[:, :k]), num_E),
                    sg.auto_WKS(torch.tensor(eigenvalues2[:k]), torch.tensor(eigenvectors2[:, :k]), num_E))
        elif signature == "HKS":
            return (sg.HKS(torch.tensor(eigenvalues1[:k]), torch.tensor(eigenvectors1[:, :k]), time_intv),
                    sg.HKS(torch.tensor(eigenvalues2[:k]), torch.tensor(eigenvectors2[:, :k]), time_intv))
