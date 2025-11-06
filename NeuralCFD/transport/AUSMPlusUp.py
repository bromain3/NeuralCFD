import torch

class AUSMPlusUpFlux:

    alpha = 3.0 / 16.0
    # Constants for low-Mach corrections
    Ku = 0.75
    Kp = 0.25

    def convective_flux(self, interface):
        # conservative fluxes: (it could be expressed with the conserved quantites vector and p)
        Fc = torch.zeros_like(interface.Q)
        Fc[0] = interface.rho * interface.a
        Fc[1] = interface.rho * interface.a * interface.u
        Fc[2] = (interface.rho * (interface.e + 0.5 * interface.u ** 2) + interface.p) * interface.a #TODO: investigate the formula
        Fc[3:3 + interface.nspecies, :] = interface.Ys.T * interface.rho * interface.a
        return(Fc)

    def pressure_direction(self, interface):
        Fp = torch.zeros_like(interface.Q)
        Fp[0] = 0
        Fp[1] = 1
        Fp[2] = 0
        Fp[3:3 + interface.nspecies, :] = 0
        return(Fp)

    def split_Mach(self, interface, direction=None):
        M = interface.u / interface.a
        abs_M = torch.abs(M)

        if direction == 'plus':
            Mp = torch.where(abs_M < 1, 0.25 * (M + 1) ** 2 + AUSMPlusUpFlux.alpha * (M ** 2 - 1) ** 2, 0.5 * (M + abs_M))
            return Mp
        elif direction == 'minus':
            Mm = torch.where(abs_M < 1, -0.25 * (M - 1) ** 2 - AUSMPlusUpFlux.alpha * (M ** 2 - 1) ** 2, 0.5 * (M - abs_M))
            return Mm
        else:
            raise ValueError("direction must be 'plus' or 'minus'")

    def split_Pressure(self, interface, direction=None):
        M = interface.u / interface.a
        abs_M = torch.abs(M)
        p = interface.p
        if direction == 'plus':
            Pp = torch.where(abs_M < 1, 0.25 * p * (M + 1) ** 2 * (2 - M), 0.5 * p * (1 + torch.sign(M)))
        elif direction == 'minus':
            Pp = torch.where(abs_M < 1, 0.25 * p * (M - 1) ** 2 * (2 + M), 0.5 * p * (1 - torch.sign(M)))
        else:
            raise ValueError("direction must be 'plus' or 'minus'")
        return Pp


    def add_source_terms(self, grid):
        return(0)

    def max_wave_speed(self, interfaces):
        return(torch.abs(interfaces.u) + interfaces.a)

    def AUSMPlusUp_flux(self, grid):
        L, R = grid.leftInterfaces, grid.rightInterfaces

        MLp = self.split_Mach(L, 'plus')
        MRm = self.split_Mach(R, 'minus')
        PLp = self.split_Pressure(L, 'plus')
        PRm = self.split_Pressure(R, 'minus')

        # Diffusion corrections
        delta_p = R.p - L.p
        delta_u = R.u - L.u

        M_half = MLp + MRm - AUSMPlusUpFlux.Kp * delta_p / (L.rho * L.a ** 2)
        p_half = PLp + PRm - AUSMPlusUpFlux.Ku * M_half * delta_u * L.rho * L.a

        FcL = self.convective_flux(L)
        FcR = self.convective_flux(R)
        n = self.pressure_direction(L)

        F = M_half / 2 * (FcL + FcR) - torch.abs(M_half) * (FcR - FcL) + p_half * n

        sL = self.max_wave_speed(L)
        sR = self.max_wave_speed(R)
        smax = torch.maximum(sL, sR)

        return F, torch.max(smax)

    def transport(self, dt, grid, bc):
        F_int, smax_global = self.AUSMPlusUp_flux(grid)
        grid.updateCellsState(grid.cells.Q - dt * ((grid.S[1:] * F_int[:, 1:] - grid.S[:-1] * F_int[:, :-1]) / grid.dx + self.add_source_terms(grid)), bc)

        return(smax_global)