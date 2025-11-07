import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation


def allkeys(obj):
    "Recursively find all keys in an h5py.Group."
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + allkeys(value)
            else:
                keys = keys + (value.name,)
    return keys


def completWith0s(input):
    if len(input) < 3:
        res = (3 - len(input)) * '0' + input
    else:
        res = input
    return(res)


def readCEDREDatabase(filename, caseName, type, value='Temperature'):
    #TODO: deduce type from file name (and casename also)
    with h5py.File(filename, 'r') as f:
        #print(allkeys(f))
        if type == 'histo':
            root = f'/Base/Locales/{caseName}'
            res = f[f'/Base/Locales/{caseName}']
            x = res[f'{root}/HORLOGE/DATES/Temps/ data'][()]
            y = res[f'{root}/ETATS/MOYENS/T/ data'][()]
            return x, y

        if type == 'champ':
            root = f'/Base/{caseName}#001'
            t = f[f'/Base/BaseIterativeData/TimeValues/ data'][()]
            T_s = []
            for n in range(len(t)):
                T_s.append(f[f'{root}/CHARME_VOLU#{completWith0s(str(n + 1))}/{value}/ data'][()])
            #print(allkeys(res))
            return t, np.array(T_s)

        if type == 'geom':
            root = f'/Base/{caseName}#001/.cedre#Geometry/CellCoordinates'
            x = f[f'{root}/CoordinateX/ data'][()]
            y = f[f'{root}/CoordinateY/ data'][()]
            z = f[f'{root}/CoordinateZ/ data'][()]
            return x, y, z


def loadData(dataPath, caseName):
    with open(os.path.join(dataPath, caseName), 'r') as f:
        simData = json.load(f)
    return(simData)

if __name__ == "__main__":
    value_cedre = 'RhoY_F1:P1:H2'
    value = 'rho'
    # values names : Temperature, Pression, F1:P1:H2, RhoY_F1:P1:H2, VitesseX
    t_cedre, field_cedre = readCEDREDatabase(r'/stck/bromain/cedre1d/sod_choc/archive_CHARME_tube1D_inst.cgns', 'tube1D', 'champ', value_cedre)
    x, y, z = readCEDREDatabase(r'/stck/bromain/cedre1d/sod_choc/archive_CHARME_tube1D_geom.cgns', 'tube1D', 'geom')

    sim_data = loadData('../../simulation_results/sod_choc', 'sod_choc_rus.json')

    func_field_cedre_for_interp = interp1d(t_cedre, field_cedre, axis=0, kind='linear', bounds_error=True)
    func_field_self_for_interp = interp1d(sim_data['t'], sim_data[value], axis=0, kind='linear', bounds_error=True)



    plot_times = np.linspace(sim_data['t'][0], sim_data['t'][-1], len(sim_data['t']))
    # Interpolation sur les temps de la première simulation
    field_cedre_interp = func_field_cedre_for_interp(plot_times)
    field_self_interp = func_field_self_for_interp(plot_times)

    fig, ax = plt.subplots()
    graph_cedre = ax.plot(x, field_cedre_interp[0])[0]
    graph = ax.plot(sim_data['x'], field_self_interp[0])[0]

    min_y = min([min(i) for i in field_self_interp] + [min(i) for i in field_cedre_interp])
    max_y = max([max(i) for i in field_self_interp] + [max(i) for i in field_cedre_interp])
    delta_y = max_y - min_y
    blank_proportion = 0.05
    plt.ylim(min_y - blank_proportion * delta_y, max_y + blank_proportion * delta_y)
    time = plot_times
    plt.title(value + f' | t = {time}')


    def update(frame):
        # creating a new graph or updating the graph
        graph.set_ydata(field_self_interp[frame])
        graph_cedre.set_ydata(field_cedre_interp[frame])
        time = plot_times[frame]
        plt.title(value + f' | t = {time:.6e}')


    anim = FuncAnimation(fig, update, frames=len(plot_times), interval=10)
    #plt.show()  # TODO: remove what is under
    plt.show(block=False)
    input("Animation terminée — appuie sur Entrée pour continuer...")
    plt.close(fig)
