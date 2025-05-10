import numpy as np
import torch
import tntorch as tn
from scipy.stats import multivariate_normal

num_dimensions = 2
mu = np.array([0.10, 0.10])  # Mean vector
cov_matrix = np.array([[0.20, 0.35],[ 0.16, 0.07]])  # Covariance matrix 

def make_psd(cov):
    # 1. Simmetrizza
    cov_sym = 0.5 * (cov + cov.T)
    
    # 2. Autovalori e autovettori
    eigvals, eigvecs = np.linalg.eigh(cov_sym)

    # 3. Azzeramento degli autovalori negativi
    eigvals_clipped = np.clip(eigvals, a_min=1e-6, a_max=None)  # imposta min=1e-6 per evitare problemi numerici

    # 4. Ricostruzione matrice PSD
    cov_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return cov_psd


def generate_tt_pdf(mu, cov, num_qubits_per_dim=10, max_tt_rank=100):
    n_dims = len(mu)
    grid_size = 2 ** num_qubits_per_dim
    #symmtrizziamo la matrice di covarianza
    cov_psd = make_psd(cov)

    # Griglia in ogni dimensione (intervallo centrato attorno a mu)
    domain = [torch.arange(1,m) fon n in range(n_dims)]

    def gaussian_pdf(x):
        return 1/ (((2 * np.pi) ** (n_dims) * np.linalg.det(cov_psd)) ** 0.5) 

        

    tt_tensor = tn.cross()

    
    return tt_tensor

tt = generate_tt_pdf(mu, cov_matrix)
print(tt)