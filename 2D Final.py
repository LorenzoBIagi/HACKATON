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
num_dimensions = 2 #TWO DIMENSIONAL CASE
mu = np.array([0.10, 0.10])  # Mean vector
cov_matrix = np.array([[0.20, 0.35],[ 0.16, 0.07]])  # Covariance matrix 
#Make it positive semidefinite
cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  
cov_matrix = cov_matrix @ cov_matrix
def gaussian(x):
    return 1/ (((2 * np.pi) ** (num_dimensions) * np.abs(np.linalg.det(cov_matrix))) ** 0.5) * np.exp(-0.5 * ((x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu)))

#DISCRETIZATION PARAMETERS
q=10 #qubits
d=q//num_dimensions #samples in each direction (MUST BE MULTIPLE)
N = 2**(2*d) 

# DISCRETIZATION
vectroized_function = [] 

for x in np.linspace(mu[0] - 3*np.sqrt(cov_matrix[0,0]), mu[0] + 3*np.sqrt(cov_matrix[0,0]), 2**d):
    for y in np.linspace(mu[1] - 3*np.sqrt(cov_matrix[1,1]), mu[1] + 3*np.sqrt(cov_matrix[1,1]), 2**d):
        vectroized_function.append(gaussian(np.array([x,y])))



xs = list(range(N))
ys = vectroized_function

# Bin into 100 columns
num_bins = 100
binned_sums = np.add.reduceat(ys, np.linspace(0, N, num_bins+1, dtype=int)[:-1])

# X-axis as bin indices
xs_binned = np.arange(num_bins)

# PLOT OF DISTRIBUTION
plt.figure(figsize=(12, 4))
plt.bar(xs_binned, binned_sums, width=1.0)
plt.ylabel('Summed Probability')
plt.xlabel('Bin')
plt.title('Probability distribution (2D)')
plt.show()

#CONVERSION TO TENSOR NETWORK

vectroized_function = np.array(vectroized_function) 

#RESHAPING

shape = (2,)*(d*num_dimensions)         # (2,2,...,2,2)
A = vectroized_function.reshape(shape)  #tensor nump reshaped
T=tn.Tensor(A)                          #tensor torch

#TENSOR TRAIN WITH TT-CROSS

TTrain = tn.cross(
    function=lambda x: x,   
    tensors=[T],                          
    ranks_tt=16,                 # Forcing maximum virtual dimension (Increased to better see the pattern in the plot)
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
    
    
    
    tronc = int(min(2**(min(k+1, int(np.log2(nk[k])+1))),nk[k+1]))
    R = R[:tronc, :]
    
    

    if k != len(cores) - 1:
        cores[k+1] = np.tensordot(R, cores[k+1],axes=([-1], [0]))

    # qubit action
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


# Simulation
simulator = AerSimulator()
compiled = transpile(qc, simulator)
job = simulator.run(compiled, shots=50024)
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

num_bins = 100
binned_sums = np.add.reduceat(ys, np.linspace(0, N, num_bins+1, dtype=int)[:-1])

# X-axis as bin indices
xs_binned = np.arange(num_bins)

# TENSOR NETWORK PLOT
plt.figure(figsize=(12, 4))
plt.bar(xs_binned, binned_sums, width=1.0)
plt.ylabel('Counts')
plt.xlabel('State (decimal)')
plt.title('Tensor Network Representation (2D)')
plt.show()