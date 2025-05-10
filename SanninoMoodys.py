import numpy as np
import torch
import tntorch as tn
from scipy.stats import multivariate_normal

num_dimensions = 2
mu = np.array([0.10, 0.10])  # Mean vector
cov_matrix = np.array([[0.20, 0.35],[ 0.16, 0.07]])  # Covariance matrix 


def generate_tt_pdf(mu, cov, num_qubits_per_dim=10, max_tt_rank=8):
    n_dims = len(mu)
    grid_size = 2 ** num_qubits_per_dim

    # Griglia in ogni dimensione (intervallo centrato attorno a mu)
    domain_np = [np.linspace(mu[d] - 3*np.sqrt(cov[d,d]), mu[d] + 3*np.sqrt(cov[d,d]), grid_size) for d in range(n_dims)]
    domain = [torch.tensor(d, dtype=torch.float32) for d in domain_np]

    # Oggetto scipy multivariate normal
    dist = multivariate_normal(mean=mu, cov=cov)

    # Funzione PDF vettorializzata
    def pdf_function(*args):
        stacked = torch.stack(args, dim=-1).numpy()  # shape (N, D)
        pdf_vals = dist.pdf(stacked)
        return torch.tensor(pdf_vals, dtype=torch.float32)

    # Approssimazione con TT-cross con rango massimo specificato
    tt_tensor = tn.cross(function=pdf_function, domain=domain, max_rank=max_tt_rank)
    return tt_tensor

tt = generate_tt_pdf(mu, cov_matrix)
print(tt)