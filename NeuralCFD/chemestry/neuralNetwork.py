import POLENE.PYTHONexport as PYTHONexport
import NeuralCFD.utilities_functions.primitive_converter as primitive_converter
import torch

class neuralNetwork:

    def __init__(self, modelPath, device, dtype, thermoSpecies=None):
        self.modelPath = modelPath
        self.cols, self.models, self.clusterer = PYTHONexport.loadPYTHONexportFolder(modelPath)

        self.device = device
        self.dtype = dtype

        tidy_cols = list(self.cols)
        tidy_cols = [e for e in tidy_cols if e not in ('T', 'P')]
        if thermoSpecies is None:
            raise ValueError('You should give thermo object columns to be sure it is ordered as the neural network expects it') # TODO: transform it as a warning
        elif not thermoSpecies == tidy_cols: raise ValueError('model inputs and thermo not coherent')
        # changing dtype of the model to double precision for the moment as it is what numpy uses
        for i in self.models:
            i.model.to(dtype)
            i.model.eval()
            i.to(device)

    def computeTPYWithNetwork(self, grid, dt):  # TODO: optimize memory usage
        # T, p: shape (ncells,), Ys: shape (ncells, nspecies)
        #TODO: check if the network takes P in input
        input = torch.hstack((grid.cells.T[:, None], grid.cells.p[:, None], grid.cells.Ys))
        clust = torch.tensor(self.clusterer.computeClusterIndexTorch(input), device=self.device, dtype=self.dtype)

        # 1. trier les indices par cluster
        sorted_idx = torch.argsort(clust)
        sorted_clust = clust[sorted_idx]

        results = torch.zeros_like(input)

        # 2. trouver les frontières de changement de cluster
        boundaries = torch.where(sorted_clust[1:] != sorted_clust[:-1])[0] + 1
        splits = torch.tensor_split(sorted_idx, boundaries.tolist())
        # 3. boucle cluster par cluster
        for idx_block in splits:
            cid = int(clust[idx_block[0]].item())  # cluster id pour ce bloc
            X_cluster = input[idx_block]
            y_pred = self.models[cid].inputScaler.transformTorch(X_cluster)
            y_pred = self.models[cid](y_pred)
            y_pred = self.models[cid].outputScaler.inverse_transformTorch(y_pred)

            results[idx_block] = y_pred

        T_new = results[:, 0].flatten()
        p_new = results[:, 1].flatten()
        Ys_new = results[:, 2:]

        return T_new, p_new, Ys_new

    def chemistry(self, dt, grid, thermo, bc, reactorModel='cst_p_const_rhou'):  # TODO: reactorModel should be in the neural net metadata in the very end
        if reactorModel == 'cst_v':
            # chemistry acts locally cell-by-cell. We keep rho and rho*u constant, update E and Ys
            # Call the NN: it should accept arrays and return arrays
            T_new, p_new, Ys_new = self.computeTPYWithNetwork(grid, dt)
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

        elif reactorModel == 'cst_p_const_u':
            # chemistry acts locally cell-by-cell. We keep P and u constant, update rho, E and Ys
            # Call the NN: it should accept arrays and return arrays
            T_new, p_new, Ys_new = self.computeTPYWithNetwork(grid, dt)
            p_new = grid.cells.p
            # Sanity shapes
            assert T_new.shape == grid.cells.T.shape
            assert p_new.shape == grid.cells.p.shape
            assert Ys_new.shape == grid.cells.Ys.shape
            # Reconstruct conserved: keep p and u, update rho, rho*E and rho*Y
            Rspec_new = thermo.thermoProp.R_spec_from_Y(Ys_new)
            rho = p_new / (Rspec_new * T_new)
            u = grid.cells.u

        elif reactorModel == 'cst_p_const_rhou':  # most physical and coherent with numerical spliting
            # chemistry acts locally cell-by-cell. We keep P and u constant, update rho, E and Ys
            # Call the NN: it should accept arrays and return arrays
            T_new, p_new, Ys_new = self.computeTPYWithNetwork(grid, dt)
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

