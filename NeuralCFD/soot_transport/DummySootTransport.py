import torch
from abc import ABC, abstractmethod


#TODO: try a monolithic solver where operators act as source terms
#TODO: add a function to return the max allowable time step as with geometric source terms, the CFL condition should not apply. the CFL condition must be hand derived
#TODO: add a model for curvilinear 1D solver for having the most complet solver
class DummySootTransport(ABC):

    @abstractmethod
    def numerical_flux(self, leftInterface, rightInterface):
       raise NotImplementedError('Any numerical flux class inheriting from DummyTransport must implement a numerical_flux(grid) method.')


    def add_source_terms(self, grid):
        return 0

    def max_wave_speed(self, interfaces):
        return(torch.abs(interfaces.u))

    def transport(self, dt, grid, bc):
        F_int = self.numerical_flux(grid.leftInterfaces, grid.rightInterfaces)
        grid.updateCellsState(grid.cells.Q + dt * ( - (grid.S[1:] * F_int[:, 1:] - grid.S[:-1] * F_int[:, :-1]) / grid.V + self.add_source_terms(grid)), bc)

        #TODO: think of a way to not recompute smax if it is alleredy computed (by rusanov for instance)
        smax = torch.max(torch.maximum(self.max_wave_speed(grid.leftInterfaces), self.max_wave_speed(grid.rightInterfaces)))

        return(smax)