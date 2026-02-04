import json
from NeuralCFD.utilities_functions.primitive_converter import primitive_to_conserved
import torch

def read_field(path, thermo):
    device = thermo.device
    dtype = thermo.dtype

    with open(path, 'r') as f:
        simData = json.load(f)
    x = torch.tensor(simData['x'], device=device, dtype=dtype)

    Ys = []
    for k, v in simData.items():
        if 'Y_' in k:
            Ys.append(torch.tensor(v[-1], device=device, dtype=dtype))
    Ys = torch.stack(Ys).T

    Ns = []
    for k, v in simData.items():
        if 'N_' in k:
            Ns.append(torch.tensor(v[-1], device=device, dtype=dtype))
    if Ns != []:
        Ns = torch.stack(Ns).T
    else:
        Ns = torch.empty((Ys.shape[0], 0), device=device, dtype=dtype)

    rho = torch.tensor(simData['rho'][-1], device=device, dtype=dtype)
    u = torch.tensor(simData['u'][-1], device=device, dtype=dtype)
    p = torch.tensor(simData['P'][-1], device=device, dtype=dtype)
    T = torch.tensor(simData['T'][-1], device=device, dtype=dtype)

    U = primitive_to_conserved(rho, u, p, T, Ys, Ns, thermo)

    return U, x