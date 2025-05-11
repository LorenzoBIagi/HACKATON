import numpy as np
import torch
import tntorch as tn
from scipy.stats import multivariate_normal

def gaussian(x):
    return 1/ (((2 * np.pi) ** (num_dimensions) * np.abs(np.linalg.det(cov_matrix))) ** 0.5) * np.exp(-0.5 * ((x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu)))

num_dimensions = 4
mu = np.array([0.10, 0.10, 0.23, 0.17])  # Mean vector
cov_matrix = np.array([[0.20, 0.35, 0.12, 0.23],[0.10, 0.28, 0.19, 0.13],[0.10, 0.20, 0.10, 0.10],[0.19, 0.03, 0.07, 0.27]])  # Covariance matrix 

#PARAMETRI DISCRETIZZAZIOINE

d = 12                # qubit
m = 2**(d//4)               # dimensioni discretizzazione



vectroized_function = []
for x in np.linspace(mu[0] - 3*np.sqrt(cov_matrix[0,0]), mu[0] + 3*np.sqrt(cov_matrix[0,0]), 2**m):
    for y in np.linspace(mu[1] - 3*np.sqrt(cov_matrix[1,1]), mu[1] + 3*np.sqrt(cov_matrix[1,1]), 2**m):
        for z in np.linspace(mu[2] - 3*np.sqrt(cov_matrix[2,2]), mu[2] + 3*np.sqrt(cov_matrix[2,2]), 2**m):
            for w in np.linspace(mu[3] - 3*np.sqrt(cov_matrix[3,3]), mu[3] + 3*np.sqrt(cov_matrix[3,3]), 2**m):
                vectroized_function.append(gaussian(np.array([x,y,z,w])))
                
vectroized_function = np.array(vectroized_function) # vettore probabilità discreta

shape = (2,)*(d*num_dimensions)         # (2,2,2,2)
A = vectroized_function.reshape(shape) #tensore numpy
T=tn.Tensor(A)      #tensore torch


TTrain = tn.cross(
    function=lambda x: x,   # identità su ciascuna fibra
    tensors=[T],            # lista di un solo tensore               # tolleranza desiderata
    rmax=8,                 # rank massimo ammesso
)


print(TTrain)


