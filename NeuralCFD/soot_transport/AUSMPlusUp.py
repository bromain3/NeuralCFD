import torch
from NeuralCFD.soot_transport.DummySootTransport import DummySootTransport

class AUSMPlusUpFlux(DummySootTransport):

    alpha = 3.0 / 16.0
    # Constants for low-Mach corrections
    Ku = 0.75
    Kp = 0.25

    def convective_flux(self, interface):
        # conservative fluxes: (it could be expressed with the conserved quantites vector and p)
        Fc = torch.zeros_like(interface.Q)
        Fc[0] = 0.0
        Fc[1] = 0.0
        Fc[2] = 0.0
        Fc[3:3 + interface.nspecies, :] = 0.0
        Fc[3 + interface.nspecies:3 + interface.nspecies + interface.nclasses, :] = interface.ns.T * interface.a
        return(Fc)

    def pressure_direction(self, interface):
        Fp = torch.zeros_like(interface.Q)
        Fp[0] = 0.0
        Fp[1] = 0.0
        Fp[2] = 0.0
        Fp[3:3 + interface.nspecies, :] = 0.0
        Fp[3 + interface.nspecies:3 + interface.nspecies + interface.nclasses, :] = 0.0
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

    def numerical_flux(self, leftInterfaces, rightInterfaces):

        MLp = self.split_Mach(leftInterfaces, 'plus')
        MRm = self.split_Mach(rightInterfaces, 'minus')
        PLp = self.split_Pressure(leftInterfaces, 'plus')
        PRm = self.split_Pressure(rightInterfaces, 'minus')

        # Diffusion corrections
        delta_p = rightInterfaces.p - leftInterfaces.p
        delta_u = rightInterfaces.u - leftInterfaces.u

        M_half = MLp + MRm - AUSMPlusUpFlux.Kp * delta_p / (leftInterfaces.rho * leftInterfaces.a ** 2)
        p_half = PLp + PRm - AUSMPlusUpFlux.Ku * M_half * delta_u * leftInterfaces.rho * leftInterfaces.a

        FcL = self.convective_flux(leftInterfaces)
        FcR = self.convective_flux(rightInterfaces)
        n = self.pressure_direction(leftInterfaces)  # the chosen interface does not matter, it is just for output size

        F = M_half / 2 * (FcL + FcR) - torch.abs(M_half) * (FcR - FcL) + p_half * n
        #TODO: check if adding p just on the right row is more efficient instead of adding a full vector

        return F