import torch
from scipy.stats import gaussian_kde
import numpy as np

from numpy.typing import NDArray



def compute_jsd(f1: np.array, f2: np.array) -> float:
    """
    Computes the Jensen-Shannon Divergence (JSD) between two arrays.
    
    Parameters:
    - f1: First input array of feature.
    - f2: Second input array  of feature.
    
    Returns:
    - Adjusted Jensen-Shannon Divergence (1 - JSD) between KDEs of f1 and f2.
    """
    # Remove NaNs and Infs
    f1_clean = f1[np.isfinite(f1)]
    f2_clean = f2[np.isfinite(f2)]
    
    # Compute KDEs
    kde1 = gaussian_kde(f1_clean)
    kde2 = gaussian_kde(f2_clean)
    
    # Determine common range for evaluation
    x_min = min(min(f1_clean), min(f2_clean))
    x_max = max(max(f1_clean), max(f2_clean))
    num_points = max(f1_clean.shape[0], f2_clean.shape[0])
    x = np.linspace(x_min, x_max, num_points)
    
    # Evaluate KDEs on the common range
    d1 = kde1(x)
    d2 = kde2(x)
    
    # Compute KL Divergence
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
    # Compute Jensen-Shannon Divergence
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
    # Adjust the KDEs to avoid division by zero or log(0) and normalize
    epsilon = 1e-10
    d1 += epsilon
    d2 += epsilon
    d1 /= np.sum(d1)
    d2 /= np.sum(d2)
    
    # Compute adjusted JSD
    jsd = js_divergence(d1, d2)
    jsd_adj =  jsd
    
    return jsd_adj



def compute_mapping_metric(C, D, eigenvalues1, eigenvalues2, k):
    isoC = torch.mm(C, torch.diag(eigenvalues1[:k])) - torch.mm(torch.diag(eigenvalues2[:k]), C)
    isoD = torch.mm(D, torch.diag(eigenvalues2[:k])) - torch.mm(torch.diag(eigenvalues1[:k]), D)
    inverDC = torch.mm(D, C) - torch.eye(k, device=C.device)
    inverCD = torch.mm(C, D) - torch.eye(k, device=C.device)
    harmitianCD = torch.mm(C, D.T) - torch.eye(k, device=C.device)
    harmitianDC = torch.mm(D, C.T) - torch.eye(k, device=C.device)
    
    return ( 10 * isoC.norm()**2 + 10 * isoD.norm()**2 + inverDC.norm()**2 + inverCD.norm()**2 + harmitianCD.norm()**2 +  harmitianDC.norm()**2).sqrt().item()



def compute_Fmap_distance(feature1: NDArray, feature2: NDArray, k: int, C: NDArray, D: NDArray, eigenvectors1: NDArray, eigenvectors2: NDArray) -> float:
        A = torch.mv(torch.tensor(eigenvectors1[:, :k].T, dtype=torch.float32), feature1)
        B = torch.mv(torch.tensor(eigenvectors2[:, :k].T, dtype=torch.float32), feature2)
        diffC = C @ A - B
        diffD = D @ B - A
        C_pr = abs(torch.linalg.det(C)) - 1
        D_pr = abs(torch.linalg.det(D)) - 1
        distance = ((diffC.norm()**2 + diffD.norm()**2)/k).sqrt().item() + + ((C_pr.norm()**2 + D_pr.norm()**2)/k).sqrt().item()
        return distance
    
    
def compute_all_distances(feature1: NDArray, feature2: NDArray, k: int, C: NDArray, D: NDArray, eigenvectors1: NDArray, eigenvectors2: NDArray) -> list:
    distances = []
    Fmap_dist = compute_Fmap_distance(feature1, feature2, k, C, D, eigenvectors1, eigenvectors2)
    jsd_dist = compute_jsd(feature1.cpu().numpy(), feature1.cpu().numpy())
    distances.append([Fmap_dist, jsd_dist])
    return distances
