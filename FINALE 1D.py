import numpy as np
import torch
import tntorch as tn
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate

%matplotlib inline


#PARAMETERS OF GAUSSIAN DISTRIBUTION
num_dimensions = 1 #ONE DIMENSIONAL CASE
mu = 0.10   # Mean vector
cov_matrix = 0.20  # Covariance matrix 

#DISCRETIZATION PARAMETERS

d = 10                 # qubit dimensions
m = 2**d               # "physical" dimensions

#DISCRETIZATION

domain_np = np.linspace(mu - 3*np.sqrt(cov_matrix), mu + 3*np.sqrt(cov_matrix), m)
def gaussian(x):
    return (1/(np.sqrt(2*np.pi*cov_matrix)))*np.exp(-0.5*((x-mu)**2)/np.sqrt(cov_matrix))

vec = [gaussian(x) for x in domain_np]    

xs = list(range(m))
ys = vec

# PLOT OF DISTRIBUTION
plt.figure(figsize=(12, 4))
plt.bar(xs, ys, width=1.0)
plt.ylabel('Counts')
plt.xlabel('State (decimal)')
plt.title('Probability distribution')
plt.show()


#CONVERSION TO TENSOR NETWORK

vec =  np.array(vec)    # Discrete probability vector

shape = (2,)*d         # (2,2,2,2,..,2)
A = vec.reshape(shape) #tensor nump reshapedy
T=tn.Tensor(A)      #tensor torch


#TENSOR TRAIN WITH TT-CROSS

TTrain = tn.cross(
    function=lambda x: x,   
    tensors=[T],            
    ranks_tt=8,           # Forcing maximum virtual dimension
)



print(TTrain)


#FROM TENSOR TRAIN TO CIRCUIT
#Cores counting

cores_torch = TTrain.cores
cores = [c.cpu().numpy() for c in cores_torch]

#IMPLEMENTATION ALGORITHM FROM PAPER

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
    cores[k] = np.reshape(cores[k], (j * i, n_k))

    
    U, S, V = np.linalg.svd(cores[k])
    

    S_prime = np.zeros_like(cores[k])
    for i in range(len(S)):
        S_prime[i,i] = S[i]
    

    R = S_prime @ V
    
    
    print("R prima",R.shape)
    tronc = int(min(2**(min(k+1, int(np.log2(nk[k])+1))),nk[k+1]))
    R = R[:tronc, :]
    
    print("R  dopo",R.shape)

    if k != len(cores) - 1:
        cores[k+1] = np.tensordot(R, cores[k+1],axes=([-1], [0]))

    # qubit aciton
    start = k + 1
    mn    = min(start, int(np.log2(nk[k])))
    diff  = int(start - mn)
    qubits = list(range(diff-1, start))
    
    U_list = U.tolist()
    W.append([U_list, qubits])

# Control printing
for idx, (U_list, qubits) in enumerate(W):
    print(f"Unitary {idx}: acts on qubits {qubits}, matrix with dimension {len(U_list)}")


#CIRCUIT PARAMETERS

n_qubits = 10
W = W[::-1]
gates =W

# CIRCUIT CREATION
qc = QuantumCircuit(n_qubits,n_qubits)


for gate in gates:
    qc.append(UnitaryGate(gate[0]), gate[1][::-1])

qc.measure(range(n_qubits)[::-1], range(n_qubits))
simulator = AerSimulator()
compiled = transpile(qc, simulator)

# Simulation
job = simulator.run(compiled, shots=57024)
result = job.result()

# Results
qc.draw('mpl')
counts_bin = result.get_counts()

# bin to dec conversion
counts_dec = {int(bstr, 2): cnt for bstr, cnt in counts_bin.items()}
n = qc.num_qubits
N = 2**n
xs = list(range(N))
ys = [counts_dec.get(x, 0) for x in xs]

# TENSOR NETWORK PLOT
plt.figure(figsize=(12, 4))
plt.bar(xs, ys, width=1.0)


plt.ylabel('Counts')
plt.xlabel('StatE (decimal)')
plt.title('Tensor Network Representation')
plt.show()