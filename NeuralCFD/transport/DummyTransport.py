import torch
from abc import ABC, abstractmethod

class DummyTransport(ABC):

    @abstractmethod
    def numerical_flux(self, leftInterface, rightInterface):
       raise NotImplementedError('Any numerical flux class inheriting from DummyTransport must implement a numerical_flux(grid) method.')

    def add_source_terms(self, grid):
        return(0)

    def max_wave_speed(self, interfaces):
        return(torch.abs(interfaces.u) + interfaces.a)


    def transport(self, dt, grid, bc):
        F_int = self.numerical_flux(grid.leftInterfaces, grid.rightInterfaces)
        grid.updateCellsState(grid.cells.Q - dt * ((grid.S[1:] * F_int[:, 1:] - grid.S[:-1] * F_int[:, :-1]) / grid.dx + self.add_source_terms(grid)), bc)

        #TODO: think of a way to not recompute smax if it is alleredy computed (by rusanov for instance)
        smax = torch.max(torch.maximum(self.max_wave_speed(grid.leftInterfaces), self.max_wave_speed(grid.rightInterfaces)))

        return(smax)