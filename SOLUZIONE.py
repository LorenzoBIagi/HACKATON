import numpy as np
import tntorch as tn
import torch

#Dati 1D
num_dimensions = 1
mu = np.array(0.10)   # Mean vector
cov_matrix = np.array(0.20)  # Covariance matrix 

cov_sym = 0.5 * (cov_matrix + cov_matrix.T)
def function(x,mu,cov_sym):
    n = x.shape[0]
    determinant = np.linalg.det(cov_sym)
    return 1/((2*np.pi)**(n/2) *determinant) * np.exp(-0.5 * (x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu))

domain_np = [np.linspace(mu[d] - 3*np.sqrt(cov_sym[d,d]), mu[d] + 3*np.sqrt(cov_sym[d,d]), grid_size) for d in range(x.shape[0])]
    

t = tn.cross(function=function, domain=domain_np, max_rank=8)
print(t)


