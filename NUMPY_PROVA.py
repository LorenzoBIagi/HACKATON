import numpy as np
import torch
import tntorch as tn
from tntorch.cross import cross

#PARAMETRI GAUSSIANA

num_dimensions = 1
mu = 0.10   # Mean vector
cov_matrix = 0.20  # Covariance matrix 

#PARAMETRI DISCRETIZZAZIOINE

d = 10                 # qubit
m = 2**d               # dimensioni discretizzazione

#GAUSSIANA DISCRETA

domain_np = np.linspace(mu - 3*np.sqrt(cov_matrix), mu + 3*np.sqrt(cov_matrix), m)
def gaussian(x):
    return 1/(np.sqrt(2*np.pi*cov_matrix))*np.exp(-0.5*((x-mu)**2)/np.sqrt(cov_matrix))



vec =  np.array([gaussian(x) for x in domain_np])    # vettore probabilità discreta

shape = (2,)*d         # (2,2,2,2)
A = vec.reshape(shape) #tensore numpy
T=tn.tensor(A)      #tensore torch


#CREAZIONE TENSOR TRAIN

TTrain = tn.cross(
    function=lambda x: x,   # identità su ciascuna fibra
    tensors=[T],            # lista di un solo tensore
    eps=1e-6,               # tolleranza desiderata
    rmax=8,                # rank massimo ammesso
    verbose=True
)

print(T)
print(A[0,0,1,1])  
