import NeuralCFD.post_processing.post_processing as pp


import matplotlib
import matplotlib.pyplot as plt


from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from cycler import cycler


N = 7

cmap = plt.get_cmap("plasma")
colors = [cmap(i / (N - 1)) for i in range(N)]


# 3. Créer un cycler propre
linestyles = ["-", "--", "-.", ":", "-", "--", "-."]


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


post_proc = pp.PostProcessing('data', 'auto_allumage_full_cantera.json')
fields = ['auto_allumage_full_cantera.json',
          'auto_allumage_full_neural.json',
          'auto_allumage_reduced_neural.json',
          'auto_allumage_reduced_neural_approx_thermo.json',
          'auto_allumage_DRGEP_35_cantera.json',
          'auto_allumage_DRGEP_24_cantera.json',
          'auto_allumage_SAR_30_cantera.json',
          'auto_allumage_SAR_28_cantera.json',
          'auto_allumage_SAR_15_cantera.json']
#post_proc.ISOCStats(fields, baseline='auto_allumage_full_cantera.json')
#post_proc.compareFieldsEvolution(['map_full_network_200/T_1624.8484848484848_phi_0.8047474747474748.json', 'map_full_cantera_200/T_1624.8484848484848_phi_0.8047474747474748.json'], 'Y_C2H2', 0)
#post_proc.compareFieldsEvolution(['auto_allumage_full_cantera.json', 'auto_allumage_reduced_neural.json', 'auto_allumage_reduced_neural_approx_thermo.json'], 'T', 0)
post_proc.compareFieldsEvolutionISOC(fields, 'Y_C2H2', 0)
#post_proc.animateField('T')
#post_proc.plotVariable('T', 0)
#post_proc.animateField('P')
#post_proc.plotVariable('P', 0)
#post_proc.animateField('Y_C2H2')
#post_proc.plotVariable('Y_C2H2', 0)
