import torch
from abc import ABC, abstractmethod

class DummyTransport(ABC):

    @abstractmethod
    def numerical_flux(self, leftInterface, rightInterface):
       raise NotImplementedError('Any numerical flux class inheriting from DummyTransport must implement a numerical_flux(grid) method.')

    def add_source_terms(self, grid):
        """
        Retourne le vecteur des termes source pour chaque cellule,
        Pour prendre en compte le terme source de quantité de mouvement généré par la variation de section, on met pas défaut cette équation:
          S_mom = p * (A_{i+1/2} - A_{i-1/2}) / V_i
        """
        S = torch.zeros_like(grid.cells.Q)  # shape (nvars, ncells)
        # dA = A_{i+1/2} - A_{i-1/2}, grid.S doit être la section aux interfaces (len = ncells+1)
        dA = grid.S[1:] - grid.S[:-1]  # shape (ncells,)
        # momentum source per unit volume
        S[1] = grid.cells.p * dA / grid.V  # p_i * (A_right - A_left) / V_i
        return S

    def max_wave_speed(self, interfaces):
        return(torch.abs(interfaces.u) + interfaces.a)

    def transport(self, dt, grid, bc):
        F_int = self.numerical_flux(grid.leftInterfaces, grid.rightInterfaces)
        grid.updateCellsState(grid.cells.Q + dt * ( - (grid.S[1:] * F_int[:, 1:] - grid.S[:-1] * F_int[:, :-1]) / grid.V + self.add_source_terms(grid)), bc)

        #TODO: think of a way to not recompute smax if it is alleredy computed (by rusanov for instance)
        smax = torch.max(torch.maximum(self.max_wave_speed(grid.leftInterfaces), self.max_wave_speed(grid.rightInterfaces)))

        return(smax)