import numpy as np
import tntorch as tn
import torch

#Dati 1D
num_dimensions = 1
mu = np.array(0.10)   # Mean vector
cov_matrix = np.array(0.20)  # Covariance matrix 

bond_dimension = 8
qubits = 10
cov_sym = 0.5 * (cov_matrix + cov_matrix.T)
eigvals, eigvecs = np.linalg.eigh(cov_sym)
eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)
cov = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

grid_size = 2 ** qubits
def function(x,mu,cov):
    n = x.shape[0]
    determinant = np.linalg.det(cov)
    return 1/((2*np.pi)**(n/2) *determinant) * np.exp(-0.5 * (x-mu).T @ np.linalg.inv(cov) @ (x-mu))

domain_np = [np.linspace(mu[d] - 3*np.sqrt(cov[d,d]), mu[d] + 3*np.sqrt(cov[d,d]), grid_size) for d in range(num_dimensions)]
    

t = tn.cross(function=function, domain=domain_np, max_rank=bond_dimension)
print(t)

