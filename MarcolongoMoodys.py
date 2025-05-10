import numpy as np
import torch
import tntorch as tn
from scipy.stats import multivariate_normal


num_dimensions = 4
mu = np.array([0.10, 0.10, 0.23, 0.17])  # Mean vector
cov_matrix = np.array([[0.20, 0.35, 0.12, 0.23],[0.10, 0.28, 0.19, 0.13],[0.10, 0.20, 0.10, 0.10],[0.19, 0.03, 0.07, 0.27]])  # Covariance matrix 

#PARAMETRI DISCRETIZZAZIOINE

d = 10                 # qubit
m = 2**d               # dimensioni discretizzazione

domain_np_1 = np.linspace(mu[1] - 3*np.sqrt(cov_matrix), mu[1] + 3*np.sqrt(cov_matrix), m)
domain_np_2 = np.linspace(mu[2] - 3*np.sqrt(cov_matrix), mu[2] + 3*np.sqrt(cov_matrix), m)
domain_np_3 = np.linspace(mu[3] - 3*np.sqrt(cov_matrix), mu[3] + 3*np.sqrt(cov_matrix), m)
domain_np_4 = np.linspace(mu[4] - 3*np.sqrt(cov_matrix), mu[4] + 3*np.sqrt(cov_matrix), m)


def gaussian(x):
    return 1/ (((2 * np.pi) ** (num_dimensions) * np.abs(np.linalg.det(cov_matrix))) ** 0.5) * np.exp(-0.5 * ((x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu)))

# Lista per raccogliere i risultati
result = []

# Iterazione su tutti i valori delle variabili
for x1 in domain_np_1:
    for x2 in domain_np_2:
        for x3 in domain_np_3:
            for x4 in domain_np_4:
                # Calcolo della funzione
                valore = gaussian(x1, x2, x3, x4)
                # Aggiunta del valore alla lista
                result.append(valore)

# Conversione della lista in array unidimensionale
result_array = np.array(result)



#GAUSSIANA DISCRETA


shape = (2,)*(d*num_dimensions)         # (2,2,2,2)
A = result_array.reshape(shape) #tensore numpy
T=tn.Tensor(A)      #tensore torch


#CREAZIONE TENSOR TRAIN

TTrain = tn.cross(
    function=lambda x: x,   # identità su ciascuna fibra
    tensors=[T],            # lista di un solo tensore               # tolleranza desiderata
    rmax=8,                 # rank massimo ammesso
)


print(TTrain)
print(A[0,0,1,1])  

