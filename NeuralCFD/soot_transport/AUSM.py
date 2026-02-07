import torch
from NeuralCFD.soot_transport.DummySootTransport import DummySootTransport

class AUSMFlux(DummySootTransport):

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
            splited_M = torch.where(abs_M < 1, (M + 1) ** 2 / 4, (M + abs_M) / 2)#TODO: compare computation speed of torch.where vs (tensor < X) * tensor
        elif direction == 'minus':
            splited_M = torch.where(abs_M < 1, - (M - 1) ** 2 / 4, (M - abs_M) / 2)  # TODO: compare computation speed of torch.where vs (tensor < X) * tensor
        else:
            raise ValueError(f'Unknown direction {direction} for split mach computation')
        return splited_M

    def split_Pressure(self, interface, direction=None):
        M = interface.u / interface.a
        abs_M = torch.abs(M)
        p = interface.p
        if direction == 'plus':
            splited_P = torch.where(abs_M < 1, p * (M + 1) ** 2 / 4 * (2 - M), p * (M + abs_M) / 2 / M)#TODO: compare computation speed of torch.where vs (tensor < X) * tensor
        elif direction == 'minus':
            splited_P = torch.where(abs_M < 1, p * (M - 1) ** 2 / 4 * (2 + M), p * (M - abs_M) / 2 / M)#TODO: compare computation speed of torch.where vs (tensor < X) * tensor
        else:
            raise ValueError(f'Unknown direction {direction} for split pressure computation')
        return splited_P

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
