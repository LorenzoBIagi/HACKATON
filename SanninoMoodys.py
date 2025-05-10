import numpy as np
import torch
import tntorch as tn
from scipy.stats import multivariate_normal

num_dimensions = 2
mu = np.array([0.10, 0.10])  # Mean vector
cov_matrix = np.array([[0.20, 0.35],[ 0.16, 0.07]])  # Covariance matrix 


def gaussian(x):
    return 1/ (((2 * np.pi) ** (num_dimensions) * np.abs(np.linalg.det(cov_matrix))) ** 0.5) * np.exp(-0.5 * ((x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu)))

d=10

vectroized_function = [] #vettore vuoto
for x in np.linspace(mu[0] - 3*np.sqrt(cov_matrix[0,0]), mu[0] + 3*np.sqrt(cov_matrix[0,0]), 2**d):
    for y in np.linspace(mu[1] - 3*np.sqrt(cov_matrix[1,1]), mu[1] + 3*np.sqrt(cov_matrix[1,1]), 2**d):
        vectroized_function.append(gaussian(np.array([x,y])))

vectroized_function = np.array(vectroized_function) # vettore probabilità discreta

shape = (2,)*(d*num_dimensions)         # (2,2,2,2)
A = vectroized_function.reshape(shape) #tensore numpy
T=tn.Tensor(A)      #tensore torch

print(shape)

TTrain = tn.cross(
    function=lambda x: x,   # identità su ciascuna fibra
    tensors=[T],            # lista di un solo tensore               # tolleranza desiderata
    rmax=8,                 # rank massimo ammesso
)


print(TTrain)
print(A[0,0,1,1]) 