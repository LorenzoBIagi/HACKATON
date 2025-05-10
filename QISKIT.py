from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate

%matplotlib inline

#PARAMETRI CIRCUITO

n_qubits = 10
gates =[]

# Crea il circuito
qc = QuantumCircuit(n_qubits, n_qubits)

for gate in gates:
    qc.append(UnitaryGate(gate[0]), gate[1])


#qc.measure([0, 1], [0, 1])


# Simulatore
#simulator = AerSimulator()

# Transpile per il simulatore
#compiled = transpile(qc, simulator)

# Esegui la simulazione
#job = simulator.run(compiled, shots=1024)
#result = job.result()

# Risultati
qc.draw('mpl')
#counts = result.get_counts()
#plot_histogram(counts)
plt.show()