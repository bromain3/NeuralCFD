import POLENE.PYTHONexport as PYTHONexport
import NeuralCFD.utilities_functions.primitive_converter as primitive_converter
import torch
from NeuralCFD.chemestry.DummyChemistry import DummyChemistry


class neuralNetwork(DummyChemistry):

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

    def computeIntegratedTPY(self, grid, dt, model_type):  # TODO: optimize memory usage
        #TODO: rais e error if model type is not the model type of the network (conts_p or const_v)
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


