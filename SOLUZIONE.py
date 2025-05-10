import numpy as np
import tntorch as tn
import torch

def function(x,mu,cov_matrix):
    n = x.shape[0]
    return 1/(2*np.pi)**(n/2) * np.exp(-0.5 * (x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu)) #funzione gaussiana

     

#Dati 1D
num_dimensions = 1
mu = 0.10   # Mean vector
cov_matrix = 0.20  # Covariance matrix 

