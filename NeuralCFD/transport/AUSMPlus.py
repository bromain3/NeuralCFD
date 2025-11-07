import torch
from NeuralCFD.transport.DummyTransport import DummyTransport

class AUSMPlusFlux(DummyTransport):

    alpha = 3.0 / 16.0

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
            Mp = torch.where(abs_M < 1, 0.25 * (M + 1) ** 2 + AUSMPlusFlux.alpha * (M ** 2 - 1) ** 2, 0.5 * (M + abs_M))
            return Mp
        elif direction == 'minus':
            Mm = torch.where(abs_M < 1, -0.25 * (M - 1) ** 2 - AUSMPlusFlux.alpha * (M ** 2 - 1) ** 2, 0.5 * (M - abs_M))
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
        # UL, UR are states at left and right of interface (shape: nvars)
        # We will vectorize this in the transport step.
        M_half = self.split_Mach(leftInterfaces, 'plus') + self.split_Mach(rightInterfaces, 'minus')
        p_half = self.split_Pressure(leftInterfaces, 'plus') + self.split_Pressure(rightInterfaces, 'minus')
        FcL = self.convective_flux(leftInterfaces)
        FcR = self.convective_flux(rightInterfaces)
        n = self.pressure_direction(leftInterfaces)  # the chosen interface does not matter, it is just for output size
        F = M_half / 2 * (FcL + FcR) - torch.abs(M_half) * (FcR - FcL) + p_half * n
        #TODO: check if adding p just on the right row is more efficient instead of adding a full vector


        return(F)
