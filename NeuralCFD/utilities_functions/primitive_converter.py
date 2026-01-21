import torch

def primitive_to_conserved(rho, u, p, T, Ys, ns, thermo):
    """
    rho (ncells,), u (ncells,), p (ncells,), T (ncells,), Ys (ncells, nspecies), ns (ncells, nclasses)
    returns U shape (3+nspecies, ncells)
    """
    N = rho.size(dim=0)

    # compute internal energy per mass directly: e = h - R_spec * T (perfect gas assumption
    Rspec = thermo.thermoProp.R_spec_from_Y(Ys)
    h = thermo.thermoProp.h_mix(T, Ys)
    e = h - Rspec * T

    kinetic = 0.5 * u**2
    E = e + kinetic  # total energy per mass

    U = torch.empty((3 + thermo.thermoProp.nspecies + ns.shape[1], N), device=thermo.device, dtype=thermo.dtype)
    U[0, :] = rho
    U[1, :] = rho * u
    U[2, :] = rho * E
    U[3:3+thermo.thermoProp.nspecies, :] = (Ys.T * rho)
    U[3+thermo.thermoProp.nspecies:3 + thermo.thermoProp.nspecies + ns.shape[1], :] = ns.T
    return U