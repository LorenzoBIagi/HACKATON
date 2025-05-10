import numpy as np
import torch
import tntorch as tn

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
T=tn.Tensor(A)      #tensore torch


#CREAZIONE TENSOR TRAIN

TTrain = tn.cross(
    function=lambda x: x,   # identità su ciascuna fibra
    tensors=[T],            # lista di un solo tensore               # tolleranza desiderata
    ranks_tt=5,                 # rank massimo ammesso
)



print(TTrain)


cores_torch = TTrain.cores
cores_numpy = [c.cpu().numpy() for c in cores_torch]

print(len(cores_numpy))
print(cores_numpy[0])
print(cores_numpy[2].shape)

    
