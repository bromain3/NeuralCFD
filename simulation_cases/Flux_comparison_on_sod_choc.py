import os.path

import NeuralCFD.grid.grid
import NeuralCFD.thermo.vectorized_thermo
import NeuralCFD.solver.solver_transport as solv_trans
from NeuralCFD.transport.rusanov import rusanovFlux
from NeuralCFD.transport.AUSM import AUSMFlux
from NeuralCFD.transport.AUSMPlus import AUSMPlusFlux
from NeuralCFD.transport.AUSMPlusUp import AUSMPlusUpFlux
import NeuralCFD.post_processing.post_processing as pp
import NeuralCFD.utilities_functions.primitive_converter as pc


import torch
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

N = 4
cmap = plt.get_cmap("coolwarm")
colors = [cmap(i / (N - 1)) for i in range(N)]


# 3. Créer un cycler propre
linestyles = ["-", "-", "-", "-"]


matplotlib.rcParams.update({
    # --- Latex ---
    'text.usetex': True,

    # --- Figure ---
    'figure.figsize': (3.5, 2.5),  # en pouces, bon format pour une colonne (~90 mm)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    #'savefig.bbox': 'tight',
    'savefig.format': 'png',

    # --- Police ---
    #'font.family': 'sans-serif',
    #'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.family': 'serif', # type cmr 10 : police latex
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,

    # --- Axes ---
    'axes.linewidth': 0.8,
    #'axes.spines.top': False,
    #'axes.spines.right': False,
    'axes.grid': False,
    #'axes.labelpad': 4,
    'axes.formatter.use_mathtext': True,    # offset en mathtext (affiche ×10^{...})
    'axes.formatter.useoffset': True,       # garde l'offset, mais on forcera la forme scientifique
    'axes.formatter.limits': (-2, 2),
    'axes.axisbelow': True,  # pour que les grilles/patchs soient en dessous des lignes

    # --- Ticks ---
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,

    # --- Lignes ---
    'lines.linewidth': 0.8,
    'lines.markersize': 2.0,

    # --- Légendes ---
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    #'legend.handletextpad': 0.4,

    # --- Couleurs ---
    'image.cmap': 'coolwarm',
    'axes.prop_cycle': cycler(color=colors) + cycler(linestyle=linestyles)

})


if __name__ == "__main__":

    device = 'cpu'
    dtype = torch.float64

    torch.set_grad_enabled(False)
    #torch.cuda.memory._set_allocator_settings()
    #torch.cuda.memory._get_current_allocator()
    #torch.cuda.memory.max_memory_reserved()
    #torch.cuda.memory.max_memory_cached()
    #torch.cuda.memory.max_memory_allocated()

    thermo = NeuralCFD.thermo.vectorized_thermo.thermo(r'../input_data/gri30.yaml', device=device, dtype=dtype)

    grid = NeuralCFD.grid.grid.grid(1.0, 200, thermo, device=device, dtype=dtype)


    bc = {'type': 'closed'}

    rho0 = 1.0 + 0.0 * grid.x
    u0 = 0.0 + 0.0 * grid.x
    p0 = 101325.0 + 0.0 * grid.x
    Ys0 = torch.zeros((grid.ncells, grid.nspecies), device=device, dtype=dtype)
    Ys0[:, 0] = 1.0

    rho0[len(grid.x)//2:] = 0.125
    p0[len(grid.x) // 2:] = 0.1 * 101325.0

    Rspec0 = thermo.thermoProp.R_spec_from_Y(Ys0)
    T0 = p0 / Rspec0 / rho0

    U0 = pc.primitive_to_conserved(rho0, u0, p0, T0, Ys0, thermo)
    print(p0[0], p0[-1])
    print(T0[0], T0[-1])

    saving_path = '../simulation_results/Flux_comparison'

    transportFlux = rusanovFlux()
    #solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    #solver.solve(0.005, 100000, 0.50, 'sod_choc_rus.json')

    transportFlux = AUSMFlux()
    #solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    #solver.solve(0.005, 100000, 0.50, 'sod_choc_AUSM.json')

    transportFlux = AUSMPlusFlux()
    #solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    #solver.solve(0.005, 100000, 0.50, 'sod_choc_AUSM+.json')

    transportFlux = AUSMPlusUpFlux()
    #solver = solv_trans.solver(grid, thermo, transportFlux, bc, U0, savingInterval=10, savingPath=saving_path)
    #solver.solve(0.005, 100000, 0.50, 'sod_choc_AUSM+up.json')

    post_proc = pp.PostProcessing('', '')
    #post_proc.animateField('rho')
    post_proc.animateFields([os.path.join(saving_path, 'sod_choc_rus.json'),
                             os.path.join(saving_path, 'sod_choc_AUSM.json'),
                             os.path.join(saving_path, 'sod_choc_AUSM+.json'),
                             os.path.join(saving_path, 'sod_choc_AUSM+up.json')], 'rho')