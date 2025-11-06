import torch

def primitive_to_conserved(rho, u, p, T, Ys, thermo):
    """
    rho (ncells,), u (ncells,), p (ncells,), T (ncells,), Ys (ncells, nspecies)
    returns U shape (3+nspecies, ncells)
    """
    N = rho.size(dim=0)

    # compute internal energy per mass directly: e = h - R_spec * T
    Rspec = thermo.thermoProp.R_spec_from_Y(Ys)
    h = thermo.thermoProp.h_mix(T, Ys)
    e = h - Rspec * T

    kinetic = 0.5 * u**2
    E = e + kinetic  # total energy per mass

    U = torch.empty((3 + thermo.thermoProp.nspecies, N), device=thermo.device, dtype=thermo.dtype)
    U[0, :] = rho
    U[1, :] = rho * u
    U[2, :] = rho * E
    U[3:3+thermo.thermoProp.nspecies, :] = (Ys.T * rho)
    return U