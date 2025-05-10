import numpy as np
import tntorch as tn
import torch

def function(x,mu,cov_matrix):
    n = x.shape[0]
    determinant = np.linalg.det(cov_matrix)
    return 1/((2*np.pi)**(n/2) *determinant) * np.exp(-0.5 * (x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu))


domain=[torch.arange(1, 33) for n in range(5)]
t = tn.cross(function=function, domain=domain)

#Dati 1D
num_dimensions = 1
mu = 0.10   # Mean vector
cov_matrix = 0.20  # Covariance matrix 

