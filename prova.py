import torch
import tntorch as tn

# 1) Costruisci A: tensore binario d-dimensionale
d = 5
A = torch.randn(*([2] * d))  # oppure i tuoi valori

# 2) Avvolgi in un Tensor-Train “grezzo”
t = tn.Tensor(A)

# 3) Esegui la cross-approximation con funzione identità
tt = tn.cross(
    function=lambda x: x,   # identità su ciascuna fibra
    tensors=[t],            # lista di un solo tensore
    eps=1e-6,               # tolleranza desiderata
    rmax=50,                # rank massimo ammesso
    verbose=True
)


print(tt)
