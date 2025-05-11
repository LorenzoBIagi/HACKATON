import numpy as np
import torch
import tntorch as tn
from scipy.stats import multivariate_normal

num_dimensions = 2
mu = np.array([0.10, 0.10])  # Mean vector
#cov_matrix = np.array([[0.20, 0.35],[ 0.16, 0.07]])  # Covariance matrix 
cov_matrix = np.array([[0.02, 0.015],[ 0.015, 0.03]])  # Covariance matrix 

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


def generate_tt_pdf(mu, cov, num_qubits_per_dim=10, max_tt_rank=8):
    n_dims = len(mu)
    grid_size = 2 ** num_qubits_per_dim
    #symmtrizziamo la matrice di covarianza
    cov_psd = make_psd(cov)

    # Griglia in ogni dimensione (intervallo centrato attorno a mu)
    domain_np = [np.linspace(mu[d] - 3*np.sqrt(cov_psd[d,d]), mu[d] + 3*np.sqrt(cov_psd[d,d]), grid_size) for d in range(n_dims)]
    domain = [torch.tensor(d, dtype=torch.float32) for d in domain_np]

    

    # Oggetto scipy multivariate normal
    dist = multivariate_normal(mean=mu, cov=cov_psd)

    # Funzione PDF vettorializzata
    def pdf_function(*args):
        stacked = torch.stack(args, dim=-1).numpy()  # shape (N, D)
        pdf_vals = dist.pdf(stacked)
        return torch.tensor(pdf_vals, dtype=torch.float32)

    # Approssimazione con TT-cross con rango massimo specificato
    tt_tensor = tn.cross(function=pdf_function, domain=domain, rmax=max_tt_rank)
    
    return tt_tensor

tt = generate_tt_pdf(mu, cov_matrix)
print(tt)