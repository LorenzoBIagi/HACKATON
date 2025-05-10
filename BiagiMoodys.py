import numpy as np
import torch
import tntorch as tn
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate


#PARAMETRI CIRCUITO

n_qubits = 2


# Crea il circuito
qc = QuantumCircuit(n_qubits, n_qubits)

qc.x(0)
qc.h(1)

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