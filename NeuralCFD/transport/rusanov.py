import torch

class rusanovFlux:

    def flux(self, interfaces):
        # conservative fluxes: (it could be expressed with the conserved quantites vector and p)
        F = torch.zeros_like(interfaces.Q)
        F[0] = interfaces.rho * interfaces.u
        F[1] = interfaces.rho * interfaces.u ** 2 + interfaces.p
        F[2] = (interfaces.rho * (interfaces.e + 0.5 * interfaces.u ** 2) + interfaces.p) * interfaces.u  # rho E * u + p u
        F[3:3 + interfaces.nspecies, :] = interfaces.Ys.T * interfaces.rho * interfaces.u
        return(F)

    def add_source_terms(self, grid):
        return(0)

    def max_wave_speed(self, interfaces):
        return(torch.abs(interfaces.u) + interfaces.a)

    def rusanov_flux(self, grid):
        # UL, UR are states at left and right of interface (shape: nvars)
        # We will vectorize this in the transport step.
        FL = self.flux(grid.leftInterfaces)
        FR = self.flux(grid.rightInterfaces)
        sL = self.max_wave_speed(grid.leftInterfaces)
        sR = self.max_wave_speed(grid.rightInterfaces)
        smax = torch.maximum(sL, sR)
        # Rusanov: F = 0.5*(FL+FR) - 0.5*smax*(UR-UL)
        return(0.5 * (FL + FR) - 0.5 * smax * (grid.rightInterfaces.Q - grid.leftInterfaces.Q), torch.max(smax))

    def transport(self, dt, grid, bc):
        F_int, smax_global = self.rusanov_flux(grid)
        grid.updateCellsState(grid.cells.Q - dt * ((grid.S[1:] * F_int[:, 1:] - grid.S[:-1] * F_int[:, :-1]) / grid.dx + self.add_source_terms(grid)), bc)

        return(smax_global)