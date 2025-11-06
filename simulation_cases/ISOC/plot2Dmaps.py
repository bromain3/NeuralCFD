import NeuralCFD.post_processing.post_processing as pp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    # --- Latex ---
    'text.usetex': True,

    # --- Figure ---
    'figure.figsize': (3.5, 2.5),  # en pouces, bon format pour une colonne (~90 mm)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    #'savefig.bbox': 'tight',
    'savefig.format': 'svg',

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
    'axes.formatter.limits': (-4, 4),
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
    'axes.prop_cycle': plt.cycler('color', plt.cm.tab10.colors),

})

post_proc = pp.PostProcessing(None, None)
baseline = 'map_full_cantera_200'


fields_to_plot = [
{'model': 'map_SAR_28_200', 'var': 'T', 'metric': 'MAE%', 'title':'Error on T of SAR 28 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'},
{'model': 'map_SAR_28_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of SAR 28 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'},

{'model': 'map_reduced_network_NASA7_200', 'var': 'T', 'metric': 'MAE%', 'title': 'Error on T of reduced NN model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'},
{'model': 'map_reduced_network_NASA7_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of reduced NN model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'},

{'model': 'map_full_network_200', 'var': 'T', 'metric': 'MAE%', 'title': 'Error on T of full NN model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'},
{'model': 'map_full_network_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of full NN model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'},

{'model': 'map_SAR_15_200', 'var': 'T', 'metric': 'MAE%', 'title':'Error on T of SAR 15 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'},
{'model': 'map_SAR_15_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of SAR 15 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'},

{'model': 'map_DRGEP_35_200', 'var': 'T', 'metric': 'MAE%', 'title':'Error on T of DRGEP 35 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'},
{'model': 'map_DRGEP_35_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of DRGEP 35 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'},

{'model': 'map_DRGEP_24_200', 'var': 'T', 'metric': 'MAE%', 'title':'Error on T of DRGEP 24 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'},
{'model': 'map_DRGEP_24_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of DRGEP 24 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'},

]

#for dic in fields_to_plot:
#    post_proc.compareModelToBaseline(dic['model'], baseline, dic['var'], dic['metric'], save_fig=True, figsize=(3.5, 2.5), n_jobs=-1, title=dic['title'], xlabel=dic['xlabel'], ylabel=dic['ylabel'], colorbarLabel=dic['cbLabel'])


DRGEP24 = {'model': 'map_DRGEP_24_200', 'var': 'T', 'metric': 'MAE%', 'title': 'Error on T of DRGEP 24 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'}
NNred = {'model': 'map_full_network_200', 'var': 'T', 'metric': 'MAE%', 'title': 'Error on T of full NN model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'mean relative absolute error [\%]'}

post_proc.compareTwoModelsToBaselineISOC(DRGEP24['model'], NNred['model'], baseline, DRGEP24['var'], DRGEP24['metric'], save_fig=True,
                                 n_jobs=-1, title1=DRGEP24['title'], title2=NNred['title'], xlabel=DRGEP24['xlabel'], ylabel=DRGEP24['ylabel'],
                                 colorbarLabel=DRGEP24['cbLabel'])

DRGEP24 = {'model': 'map_DRGEP_24_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of DRGEP 24 model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'}
NNred = {'model': 'map_full_network_200', 'var': 'concentration_C2H2', 'metric': 'AREA_ERR%', 'title': r'Error on [$C_2H_2$] of full NN model', 'xlabel': r'$T_{ini}$ [K]', 'ylabel': r'$\phi_{ini}$ [---]', 'cbLabel': r'integrated concentration error [\%]'}

post_proc.compareTwoModelsToBaselineISOC(DRGEP24['model'], NNred['model'], baseline, DRGEP24['var'], DRGEP24['metric'], save_fig=True,
                                 n_jobs=-1, title1=DRGEP24['title'], title2=NNred['title'], xlabel=DRGEP24['xlabel'], ylabel=DRGEP24['ylabel'],
                                 colorbarLabel=DRGEP24['cbLabel'])