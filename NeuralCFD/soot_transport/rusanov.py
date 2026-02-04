import torch
from NeuralCFD.transport.DummyTransport import DummyTransport

class rusanovFlux(DummyTransport):

    def flux(self, interfaces):
        # conservative fluxes: (it could be expressed with the conserved quantites vector and p)
        F = torch.zeros_like(interfaces.Q)
        F[0] = 0.0
        F[1] = 0.0
        F[2] = 0.0
        F[3:3 + interfaces.nspecies, :] = 0.0
        F[3 + interfaces.nspecies: 3 + interfaces.nspecies + interfaces.nclasses, :] = interfaces.ns.T * interfaces.u
        return(F)

    def numerical_flux(self, leftInterfaces, rightInterfaces):
        # UL, UR are states at left and right of interface (shape: nvars)
        # We will vectorize this in the transport step.
        FL = self.flux(leftInterfaces)
        FR = self.flux(rightInterfaces)
        sL = self.max_wave_speed(leftInterfaces)
        sR = self.max_wave_speed(rightInterfaces)
        smax = torch.maximum(sL, sR)
        # Rusanov: F = 0.5*(FL+FR) - 0.5*smax*(UR-UL)
        return(0.5 * (FL + FR) - 0.5 * smax * (rightInterfaces.Q - leftInterfaces.Q))
