import numpy as np
import torch
import tntorch as tn
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate

%matplotlib inline


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
    return (1/(np.sqrt(2*np.pi*cov_matrix)))*np.exp(-0.5*((x-mu)**2)/np.sqrt(cov_matrix))



vec =  np.array([gaussian(x) for x in domain_np])    # vettore probabilità discreta

shape = (2,)*d         # (2,2,2,2)
A = vec.reshape(shape) #tensore numpy
T=tn.Tensor(A)      #tensore torch


#CREAZIONE TENSOR TRAIN

TTrain = tn.cross(
    function=lambda x: x,   # identità su ciascuna fibra
    tensors=[T],            # lista di un solo tensore               # tolleranza desiderata
    ranks_tt=8,                 # rank massimo ammesso
)



print(TTrain)


cores_torch = TTrain.cores
cores = [c.cpu().numpy() for c in cores_torch]

W = []
nk = []
for k in range (len(cores)):
     nk.append(len(cores[k-1][0,0,:]))
nk.append(1)
print(nk)

for k in range(len(cores)):
    # reshape & SVD 
    j, i, n_k = cores[k].shape
    lk=np.log2(nk[k])

    print("run :",k)
    print("nk",nk[k])
    print("lk",lk)


    print("core prima",cores[k].shape)
    cores[k] = np.reshape(cores[k], (j * i, n_k))

    print("core dopo",cores[k].shape)
    U, S, V = np.linalg.svd(cores[k])
    print("SVD")
    print("U : ",U.shape)
    print("S Array : ", S.shape)
    print("V : ", V.shape)

    S_prime = np.zeros_like(cores[k])
    for i in range(len(S)):
        S_prime[i,i] = S[i]
    

    print("S Dopo : ", S_prime.shape)

    R = S_prime @ V
    
    
    print("R prima",R.shape)
    tronc = int(2**(min(k+1,np.log2(nk[k+1]))))
    R = R[:tronc, :]
    
    print("R  dopo",R.shape)

    if k != len(cores) - 1:
        cores[k+1] = np.tensordot(R, cores[k+1],axes=([-1], [0]))

    # calcolo dei qubit coinvolti
    start = k + 1
    mn    = min(start, int(np.log2(nk[k])))
    diff  = int(start - mn)
    qubits = list(range(diff-1, start))
    
    U_list = U.tolist()
    W.append([U_list, qubits])

# Stampa di controllo
for idx, (U_list, qubits) in enumerate(W):
    print(f"Unitary {idx}: acts on qubits {qubits}, matrix with dimension {len(U_list)}")


#PARAMETRI CIRCUITO

n_qubits = 10
W = W[::-1]
gates =W

# Crea il circuito
qc = QuantumCircuit(n_qubits, n_qubits)



for gate in gates:
    #print(gate[0].shape)
    qc.append(UnitaryGate(gate[0]), gate[1][::-1])


qc.measure_all()


# Simulatore
simulator = AerSimulator()

# Transpile per il simulatore
compiled = transpile(qc, simulator)

# Esegui la simulazione
job = simulator.run(compiled, shots=7024)
result = job.result()

# Risultati
qc.draw('mpl')
counts = result.get_counts()
plot_histogram(counts)
plt.show()