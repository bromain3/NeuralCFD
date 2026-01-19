import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport_chemistry_subcycle as solv
import NeuralCFD.solver.solver_transport as solvtrans
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlus import AUSMPlusFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
from NeuralCFD.chemestry.canteraChem import canteraChem
from NeuralCFD.chemestry.canteraChemMultiproc import canteraChemMultiproc
from NeuralCFD.interpolation.MUSCL import MUSCL, minmod
from NeuralCFD.interpolation.firstOrderInterp import FirstOrderInterpolation
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc
import cantera as ct

import torch
import matplotlib.pyplot as plt

def nozzle_profile(
        L=10.0e-3 / 2,
        ncells=200,
        A_throat=0.6,
        A_exit=1.0,
        A_inlet=1.0,
        frac_throat=0.5,
        shape="quadratic",
        smoothness=6.0,
        plot=True,
        device="cpu",
        dtype=torch.float64
):
    """
    Génère un profil de tuyère convergente-divergente 1D.

    Parameters
    ----------
    L : float
        Longueur totale du domaine (m).
    ncells : int
        Nombre de cellules (le vecteur retourné a ncells+1 points de maille).
    A_throat : float
        Section minimale (au col).
    A_exit : float
        Section de sortie.
    A_inlet : float
        Section d'entrée.
    x_throat : float
        Position (fraction de L) du col.
    shape : str
        "linear" | "quadratic" | "gaussian"
        Type de profil.
    smoothness : float
        Contrôle la largeur du col (utilisé pour le profil "gaussian").
    plot : bool
        Si True, affiche le profil.
    device : str
        "cpu" ou "cuda".
    dtype : torch.dtype
        Type de données PyTorch.

    Returns
    -------
    x : torch.Tensor
        Coordonnées (ncells+1,)
    A : torch.Tensor
        Aire de la tuyère (ncells+1,)
    """

    # Grille
    x = torch.linspace(0.0, L, ncells + 1, device=device, dtype=dtype)

    # --- Profil de base
    if shape == "linear":
        # Convergent linéaire + divergent linéaire
        A = torch.empty_like(x)
        left_mask = (x <= L * frac_throat)
        right_mask = ~left_mask

        # partie convergente
        A[left_mask] = A_inlet - (A_inlet - A_throat) * (x[left_mask] / (L * frac_throat))
        # partie divergente
        A[right_mask] = A_throat + (A_exit - A_throat) * ((x[right_mask] - L * frac_throat) / (L - L * frac_throat))

    elif shape == "quadratic":
        # Profil lisse avec col quadratique
        # Exemple : A(x) = A_throat + (A_inlet - A_throat)*(4*(x/L-0.5)^2)
        xi = (x / L - L * frac_throat)
        A = A_throat + (A_exit - A_throat) * (xi / (0.5 - L * frac_throat)) ** 2
        # Ajustement pour garder A(0)=A_inlet, A(L)=A_exit
        A = A - (A[0] - A_inlet) * (1 - x / L)

    elif shape == "gaussian":
        # Profil inspiré du cas de Liou (AUSM+): A(x) = 1 + K*(x-x_throat)^2
        A = A_throat + (A_inlet - A_throat) * torch.exp(-smoothness * (x / L - L * frac_throat) ** 2)
        # Ajuste les extrémités
        A = (A - A.min()) / (A.max() - A.min()) * (A_exit - A_throat) + A_throat

    else:
        raise ValueError(f"Shape '{shape}' not recognized. Use 'linear', 'quadratic' or 'gaussian'.")

    # --- Affichage
    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(x, A, label=f"Profil {shape}")
        plt.axvline(L * frac_throat, color='gray', linestyle='--', label="Throat")
        plt.xlabel("x [m]")
        plt.ylabel("A(x) [m²]")
        plt.title(f"Tuyère {shape} (L={L}, A_throat={A_throat}, A_exit={A_exit})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return x, A


def phi2Y(phi):
    MMCH4 = 16.04
    MMO2 = 31.999
    O2_N2 = 3.32

    mch4_mo2_st = 1 * MMCH4 / (2 * MMO2)
    Y = {'CH4': phi * mch4_mo2_st * 2 * MMO2, 'O2': 2 * MMO2, 'N2': 2 * MMO2 * O2_N2}
    return (Y)

if __name__ == "__main__":

    device = 'cuda:1'
    dtype = torch.float64

    torch.set_grad_enabled(False)

    th_file = r'../../input_data/gri30.yaml'
    ncells = 1000


    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(th_file, device=device, dtype=dtype)
    chem = canteraChemMultiproc(th_file, ncells, 50)
    #chem = canteraChem(th_file, ncells)


    _, A = nozzle_profile(ncells=ncells,device=device, shape='linear', plot=False)
    L = float(_[-1] - _[0])

    #interpMethode = MUSCL(minmod)
    interpMethode = FirstOrderInterpolation()
    grid = NeuralCFD.grid.grid.grid(L, ncells, thermo, interpolationMethode=interpMethode, device=device, dtype=dtype, custome_S=A)

    plt.plot(grid.x.cpu(), grid.V.cpu() / grid.dx.cpu(), label="A(x)")
    #plt.axvline(x_throat, color='gray', linestyle='--', label="Throat")
    plt.xlabel("x")
    plt.ylabel("A(x)")
    plt.title(f"Profil convergent-divergent ({grid.ncells} cellules)")
    plt.legend()
    plt.grid(True)
    plt.show()


    bc = {'type': 'symetric'}
    gas = ct.Solution(thermo.thermoFilePath)

    gas.Y = phi2Y(1.0)

    p0 = 101325.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    T0 =  650.0 + (2000.0 - 650.0) * torch.exp(-( grid.x / (1.8 * 4 * 2.5e-4) ) ** 2)  # hot spot in center

    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    for spe in gas.species_names:
        Ys0[:, thermo.thermoProp.species.index(spe)] = gas.Y[gas.species_names.index(spe)] + 0.0 * grid.x
    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0
    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)

    transportFlux = rusanovFlux()
    solver = solv.solver(grid, thermo, transportFlux, chem, bc, U0, savingInterval=15, simdt=3.0e-7, savingPath='../../simulation_results/full_validation_case', reactorModel='cst_rho', initialGridTemperature=T0)
    #solver = solvtrans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=100, savingPath='../../simulation_results/full_validation_case', initialGridTemperature=T0)
    solver.solve(0.0003, 100_000, 0.95, 'res.json')


    post_proc = pp.PostProcessing('../../simulation_results/full_validation_case', 'res.json')
    post_proc.plotSnapchot('T', -1)
    post_proc.plotSnapchot('P', -1)
    post_proc.plotSnapchot('u', -1)
    post_proc.animateField('T')