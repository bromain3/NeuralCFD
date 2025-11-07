import NeuralCFD.utilities_functions.primitive_converter as primitive_converter
from abc import ABC, abstractmethod

class DummyChemistry(ABC):

    @abstractmethod
    def computeIntegratedTPY(self, grid, dt, reactor_model):
       raise NotImplementedError('Any numerical flux class inheriting from DummyChemistry must implement a computeIntegratedTPY(grid) method.')


    def chemistry(self, dt, grid, thermo, bc, reactorModel='cst_p'):  # TODO: reactorModel should be in the neural net metadata in the very end
        if reactorModel == 'cst_rho':
            # chemistry acts locally cell-by-cell. We keep rho and rho*u constant, update E and Ys
            # Call the NN: it should accept arrays and return arrays
            T_new, p_new, Ys_new = self.computeIntegratedTPY(grid, dt, reactorModel)
            # Sanity shapes
            assert T_new.shape == grid.cells.T.shape
            assert p_new.shape == grid.cells.p.shape
            assert Ys_new.shape == grid.cells.Ys.shape
            # Reconstruct conserved: keep rho and rho*u, update rho*E and rho*Y. it means no mass transfer between cells during chemestry, so constant volume
            # TODO : for consistency, we can recompute P however for the moment, the least corrections we do to the nn predictions, the least likely it is for it to do wierd shit
            # Rspec_new = vec_th.R_spec_from_Y(Ys_new)
            # p_new = rho * Rspec_new * T_new
            rho = grid.cells.rho
            u = grid.cells.u

        elif reactorModel == 'cst_p':  # most physical and coherent with numerical spliting
            # chemistry acts locally cell-by-cell. We keep P and u constant, update rho, E and Ys
            # Call the NN: it should accept arrays and return arrays
            T_new, p_new, Ys_new = self.computeIntegratedTPY(grid, dt,reactorModel)
            p_new = grid.cells.p
            # Sanity shapes
            assert T_new.shape == grid.cells.T.shape
            assert p_new.shape == grid.cells.p.shape
            assert Ys_new.shape == grid.cells.Ys.shape
            # Reconstruct conserved: keep p and u, update rho, rho*E and rho*Y
            Rspec_new = thermo.thermoProp.R_spec_from_Y(Ys_new)
            rho = p_new / (Rspec_new * T_new)
            u = grid.cells.Q[1, :] / rho

        else:
            raise ValueError('Unknown reactor model')

        U_new = primitive_converter.primitive_to_conserved(rho, u, p_new, T_new, Ys_new, thermo)
        grid.updateCellsState(U_new, bc)