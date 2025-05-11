import numpy as np
import torch
import tntorch as tn
from qiiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate

%matplotlib inline


#PARAMETRI GAUSSIANA

num_dimensions = 2
mu = np.array([0.10, 0.10])  # Mean vector
cov_matrix = np.array([[0.20, 0.35],[ 0.16, 0.07]])  # Covariance matrix 


def gaussian(x):
    return 1/ (((2 * np.pi) ** (num_dimensions) * np.abs(np.linalg.det(cov_matrix))) ** 0.5) * np.exp(-0.5 * ((x-mu).T @ np.linalg.inv(cov_matrix) @ (x-mu)))

d=5

vectroized_function = [] #vettore vuoto
for x in np.linspace(mu[0] - 3*np.sqrt(cov_matrix[0,0]), mu[0] + 3*np.sqrt(cov_matrix[0,0]), 2**d):
    for y in np.linspace(mu[1] - 3*np.sqrt(cov_matrix[1,1]), mu[1] + 3*np.sqrt(cov_matrix[1,1]), 2**d):
        vectroized_function.append(gaussian(np.array([x,y])))

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
    tronc = int(min(2**(min(k+1, int(np.log2(nk[k])+1))),nk[k+1]))
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
qc = QuantumCircuit(n_qubits,n_qubits)



for gate in gates:
    qc.append(UnitaryGate(gate[0]), gate[1][::-1])


qc.measure(range(n_qubits)[::-1], range(n_qubits))


# Simulatore
simulator = AerSimulator()

# Transpile per il simulatore
compiled = transpile(qc, simulator)

# Esegui la simulazione
job = simulator.run(compiled, shots=10024)
result = job.result()

# Risultati
qc.draw('mpl')
counts_bin = result.get_counts()
# Converte le chiavi binarie in decimali
counts_dec = {int(bstr, 2): cnt for bstr, cnt in counts_bin.items()}

n = qc.num_qubits
N = 2**n

# Reconstruiamo asse X completo e vettore Y con zero quando mancante
xs = list(range(N))
ys = [counts_dec.get(x, 0) for x in xs]
xs = np.reshape(xs, (2**(d),2**(d)))
ys = np.reshape(ys, (2**(d),2**(d)))
from mpl_toolkits.mplot3d import Axes3D  # abilita il 3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
ax.plot_surface(
    xs[0],            # asse X: indice di riga
    xs[1],            # asse Y: indice di colonna
    ys,           # asse Z: counts
    rstride=1,
    cstride=1,
    edgecolor='none',
    cmap='viridis'
)
ax.set_title('Istogramma 3D delle counts')

plt.show()

# Plotta con matplotlib “puro”
#plt.figure(figsize=(12, 4))
#plt.bar(xs, ys, width=1.0)

# Togli le tacche e le label sull’asse x
#plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

#plt.ylabel('Counts')
#plt.xlabel('Stato (decimale)')
#plt.title('Istogramma dei risultati')
#plt.show()