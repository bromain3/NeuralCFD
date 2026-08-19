import POLENE_GPU.exports.PYTHON as PYTHONexport
import NeuralCFD.utilities_functions.primitive_converter as primitive_converter
import torch
from NeuralCFD.chemestry.DummyChemistry import DummyChemistry


class neuralNetwork(DummyChemistry):

    def __init__(self, modelPath, device, dtype, thermoSpecies=None):
        self.modelPath = modelPath
        export = PYTHONexport.loadPYTHONexportFolder(modelPath)
        self.cols = export.columns
        self.model = export.model

        self.device = device
        self.dtype = dtype

        thermoSpecies.remove('bin')

        tidy_cols = list(self.cols)
        tidy_cols = [e for e in tidy_cols if e not in ('T', 'P')]
        if thermoSpecies is None:
            raise ValueError('You should give thermo object columns to be sure it is ordered as the neural network expects it')
        elif not thermoSpecies == tidy_cols:
            raise ValueError('model inputs and thermo not coherent')

        # espèces réellement prédites par le réseau (peut exclure l'espèce fictive)
        self.outputSpecies = [e for e in export.outputColumns if e not in ('T', 'P')]
        self.fictiveSpecies = export.fictiveSpecies
        self.thermoSpecies = thermoSpecies

        self.thermoSpecies = thermoSpecies
        self.nspecies_reduced = len(tidy_cols)  # nombre d'espèces hors fictive

        self.model.to(dtype)
        self.model.eval()
        self.model.to(device)

    def computeIntegratedTPY(self, grid, dt, model_type):
        # on retire la dernière colonne (espèce fictive) avant de construire l'input
        Ys_in = grid.cells.Ys[:, :self.nspecies_reduced]
        input = torch.hstack((grid.cells.T[:, None], grid.cells.p[:, None], Ys_in)).to(self.dtype)

        y_pred = self.model.inputScaler.transformTorch(input)
        y_pred = self.model(y_pred, tensorInput=True)
        y_pred = self.model.outputScaler.inverse_transformTorch(y_pred)

        T_new = y_pred[:, 0].flatten().to(grid.dtype)
        p_new = y_pred[:, 1].flatten().to(grid.dtype)

        Ys_reduced = y_pred[:, 2:].to(grid.dtype)  # (ncells, nspecies_reduced)
        Yfict = 1.0 - torch.sum(Ys_reduced, dim=1, keepdim=True)
        Yfict = torch.clamp(Yfict, min=0.0)
        Ys_new = torch.cat((Ys_reduced, Yfict), dim=1)  # bin toujours en dernier

        return T_new, p_new, Ys_new