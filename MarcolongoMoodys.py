from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.ibmq import IBMQ

# Carica l'account IBMQ
IBMQ.load_account()  # Assicurati di aver configurato il tuo account con `IBMQ.save_account('API_TOKEN')`

# Ottieni un backend reale
provider = IBMQ.get_provider(hub='ibm-q')
backend = provider.get_backend('ibmq_manila')  # Cambia con un altro backend se necessario

# Crea un circuito quantistico
qc = QuantumCircuit(2, 2)  # Due qubit e due bit classici
qc.h(0)  # Porta Hadamard sul qubit 0
qc.cx(0, 1)  # Porta CNOT, qubit 0 come controllo e 1 come target
qc.measure([0, 1], [0, 1])  # Misura entrambi i qubit nei bit classici

print("Circuit:")
print(qc)

# Transpilazione e invio al backend
transpiled_qc = transpile(qc, backend)  # Adatta il circuito al backend
qobj = assemble(transpiled_qc, backend=backend)  # Prepara il circuito per l'esecuzione

# Esegui il circuito
job = backend.run(qobj)
print("Job ID:", job.job_id())

# Ottieni i risultati
from qiskit.tools.monitor import job_monitor
job_monitor(job)

result = job.result()
counts = result.get_counts(qc)
print("Result:", counts)
