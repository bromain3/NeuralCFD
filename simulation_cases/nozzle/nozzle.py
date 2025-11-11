import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport as solv_trans
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlus import AUSMPlusFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc

import matplotlib.pyplot as plt

import torch


def nozzle_profile(
        L=1.0,
        ncells=200,
        A_throat=0.8,
        A_exit=1.2,
        A_inlet=1.0,
        x_throat=0.5,
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
        left_mask = x <= x_throat * L
        right_mask = ~left_mask

        # partie convergente
        A[left_mask] = A_inlet - (A_inlet - A_throat) * (x[left_mask] / (x_throat * L))
        # partie divergente
        A[right_mask] = A_throat + (A_exit - A_throat) * ((x[right_mask] - x_throat * L) / (L - x_throat * L))

    elif shape == "quadratic":
        # Profil lisse avec col quadratique
        # Exemple : A(x) = A_throat + (A_inlet - A_throat)*(4*(x/L-0.5)^2)
        xi = (x / L - x_throat)
        A = A_throat + (A_exit - A_throat) * (xi / (0.5 - x_throat)) ** 2
        # Ajustement pour garder A(0)=A_inlet, A(L)=A_exit
        A = A - (A[0] - A_inlet) * (1 - x / L)

    elif shape == "gaussian":
        # Profil inspiré du cas de Liou (AUSM+): A(x) = 1 + K*(x-x_throat)^2
        A = A_throat + (A_inlet - A_throat) * torch.exp(-smoothness * (x / L - x_throat) ** 2)
        # Ajuste les extrémités
        A = (A - A.min()) / (A.max() - A.min()) * (A_exit - A_throat) + A_throat

    else:
        raise ValueError(f"Shape '{shape}' not recognized. Use 'linear', 'quadratic' or 'gaussian'.")

    # --- Affichage
    if plot:
        plt.figure(figsize=(7, 4))
        plt.plot(x, A, label=f"Profil {shape}")
        plt.axvline(x_throat * L, color='gray', linestyle='--', label="Throat")
        plt.xlabel("x [m]")
        plt.ylabel("A(x) [m²]")
        plt.title(f"Tuyère {shape} (L={L}, A_throat={A_throat}, A_exit={A_exit})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return x, A



if __name__ == "__main__":

    device = 'cuda'
    dtype = torch.float64

    torch.set_grad_enabled(False)


    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'../../input_data/gri30.yaml', device=device, dtype=dtype)




    ncells = 200
    L = 1.0
    x = torch.linspace(0, L, ncells+1)
    #A = 1 + 2.2 * (x - L/2) ** 2


    A_throat = 0.5  # section minimale
    A_max = 1.0
    x_throat = 0.5  # position du col (au milieu)
    A = torch.zeros_like(x)
    # Partie convergente (gauche)
    left_mask = x <= x_throat
    A[left_mask] = A_max - (A_max - A_throat) * (x[left_mask] / x_throat)

    # Partie divergente (droite)
    right_mask = x > x_throat
    A[right_mask] = A_throat + (A_max - A_throat) * ((x[right_mask] - x_throat) / (L - x_throat))


    grid = NeuralCFD.grid.grid.grid(L, ncells, thermo, device=device, dtype=dtype, custome_S=A)

    plt.plot(grid.x.cpu(), grid.V.cpu(), label="A(x)")
    #plt.axvline(x_throat, color='gray', linestyle='--', label="Throat")
    plt.xlabel("x")
    plt.ylabel("A(x)")
    plt.title(f"Profil convergent-divergent ({len(x)} cellules)")
    plt.legend()
    plt.grid(True)
    plt.show()



    P_inflow = 10 * 101325.0
    T_inflow = 300.0
    u_inflow = 100.0

    T_in = torch.tensor([T_inflow], dtype=dtype, device=device)
    p_in = torch.tensor([P_inflow], dtype=dtype, device=device)
    u_in = torch.tensor([u_inflow], dtype=dtype, device=device)

    Ys_in = torch.zeros((1, grid.nspecies), device=device, dtype=dtype)
    Ys_in[:, 0] = 1.0


    Rspec_in = thermo.thermoProp.R_spec_from_Y(Ys_in)
    rho_in = p_in / Rspec_in / T_in

    U_in = pc.primitive_to_conserved(rho_in, u_in, p_in, T_in, Ys_in, thermo).flatten()



    bc = {'type': 'flame1d_P_out_eq_P_in', 'inflow_state':U_in}

    P = 101325.0 #* 0.6784
    T = 300.0
    u = 0.0

    T0 = T + 0.0 * grid.x
    u0 = u + 0.0 * grid.x
    p0 = P + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    Ys0[:, 0] = 1.0

    #T0[grid.x <= L/2] = T_in
    #p0[grid.x <= L/2] = p_in
    #u0[grid.x <= L/2] = u_in

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    rho0 = p0 / Rspec0 / T0

    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)


    transportFlux = rusanovFlux()
    solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath='../../simulation_results/nozzle')
    solver.solve(0.002, 100000, 0.9, 'res.json')


    post_proc = pp.PostProcessing('../../simulation_results/nozzle', 'res.json')
    #post_proc.plotSnapchot('P', 0)
    post_proc.animateField('T')