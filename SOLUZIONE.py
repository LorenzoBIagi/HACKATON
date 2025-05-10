import numpy as np
import tntorch as tn
import torch

#Dati 1D
num_dimensions = 1
mu = 0.10   # Mean vector
cov_matrix = 0.20  # Covariance matrix 

def function(x,mu,cov_matrix):
    cov = 0.5 * (cov_matrix + cov_matrix.T)
    n = x.shape[0]
    determinant = np.linalg.det(cov)
    return 1/((2*np.pi)**(n/2) *determinant) * np.exp(-0.5 * (x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu))

domain_np = [np.linspace(mu[d] - 3*np.sqrt(cov_sym[d,d]), mu[d] + 3*np.sqrt(cov_sym[d,d]), grid_size) for d in range(x.shape[0])]
    

domain=domain()
t = tn.cross(function=function, domain=domain)



