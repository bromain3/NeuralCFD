import numpy as np
import yaml

#TODO: improve sanity checks and robustness for model selection
# divide in classes so that every model selection, ranges, coefs, etc is done at init time, so at runtime we juste have pur sheer computation done

R = 8.314_462_618_153_24  # SI
Pref = 1e5


class MM:
    O = 15.999e-3
    N = 14.0067e-3
    Ar = 39.948e-3
    H = 1.00784e-3
    C = 12.011e-3

class yamlParser:
    def __init__(self, YAML_file):
        self.Yaml_file = YAML_file
        with open(self.Yaml_file, 'r') as f:
            self.yaml_data = yaml.load(f, Loader=yaml.BaseLoader)
        self.thermo = self.getThermoData()

    def getThermoData(self):
        res = []
        for spe in self.yaml_data['species']:
            res.append({'name': spe['name'], 'composition': [(int(v), k) for k,v in spe['composition'].items()],
                        'modelType': spe['thermo']['model'], 'trange': [float(k) for k in spe['thermo']['temperature-ranges']],
                        'data': [[float(k) for k in spe['thermo']['data'][i]] for i in range(len(spe['thermo']['data']))]})
        return(res)

    def populateThermoObject(self, th):
        for i in self.thermo:
            th.AddSpeciePorperty(i['name'], i['composition'], i['modelType'], TRange=i['trange'], data=i['data'])

    def populateVectorizedThermoObject(self):
        species_names = [v['name'] for v in self.thermo]
        Tmin = [v['trange'][0] for v in self.thermo]
        Tmid = [v['trange'][1] for v in self.thermo]
        Tmax = [v['trange'][2] for v in self.thermo]
        coefs_low = [v['data'][0] for v in self.thermo]
        coefs_high = [v['data'][1] for v in self.thermo]
        molar_masses = [sum([number * getattr(MM, atom) for number, atom in v['composition']]) for v in self.thermo]
        return(ThermoData_vectorized(species_names, Tmin, Tmid, Tmax, coefs_low, coefs_high, molar_masses))


class ThermoData:
    ######## something has to be done about interpolation ranges
    # This class is far form being optimized. This issue should be adressed because the downgrade of performances caused by it is huge
    # Can be done by precomputing values and using linear interpolation, or by doing the computation in a subroutine in C or Fortran
    MODEL = 'Model'
    DATA = 'data'
    TRANGE = 'TRange'
    LOWTDATA = 'Low T data'
    HIGHTDATA = 'High T data'
    MolarMass = 'MolarMass'

    def __init__(self):
        self.__Species = {}

    def AddSpeciePorperty(self, SpecieName, Composition, ModelType, **kwargs):
        if ModelType == 'NASA7':
            if 'TRange' not in kwargs or 'data' not in kwargs:
                msg = 'The temperature range of data or data itself is missing'
                raise ValueError(msg)

            self.__Species[SpecieName] = {ThermoData.MODEL: 'NASA7',
                                          ThermoData.MolarMass: sum(
                                              [number * getattr(MM, atom) for number, atom in Composition]),
                                          ThermoData.TRANGE: kwargs[ThermoData.TRANGE],
                                          ThermoData.LOWTDATA: np.array(kwargs[ThermoData.DATA][0]),
                                          ThermoData.HIGHTDATA: np.array(kwargs[ThermoData.DATA][1])}
        else:
            msg = 'The thermodynamics model named {} is not understood'.format(ModelType)
            raise KeyError(msg)

    @property
    def Species(self):
        return (self.__Species)

    def cp(self, SpecieName, T, P):
        # Independant of P for ideal gas
        Tmin, Tmid, Tmax = self.__Species[SpecieName][ThermoData.TRANGE]
        coef_low = self.__Species[SpecieName][ThermoData.LOWTDATA][:5]
        coef_high = self.__Species[SpecieName][ThermoData.HIGHTDATA][:5]

        Temp = np.vstack([np.ones_like(T), T, T**2, T**3, T**4])  # shape (5, len(T))

        # broadcast coefficients en fonction de la température
        coef = np.where(T <= Tmid, coef_low[:, None], coef_high[:, None])
        cp_val = R * np.sum(coef * Temp, axis=0)
        return cp_val

    def cv(self, SpecieName, T, P):
        return (self.cp(SpecieName, T, P) - R)

    def h(self, SpecieName, T, P):
        # dH/dP = V (1-alpha*T) = 0 for perfect gaz as alpha = 1/T

        Tmin, Tmid, Tmax = self.__Species[SpecieName][ThermoData.TRANGE]
        coef_low = self.__Species[SpecieName][ThermoData.LOWTDATA][:6]
        coef_high = self.__Species[SpecieName][ThermoData.HIGHTDATA][:6]

        Temp = np.vstack([np.ones_like(T), T / 2, T**2 / 3, T**3 / 4, T**4 / 5, 1 / T])  # shape (5, len(T))

        coef = np.where(T <= Tmid, coef_low[:, None], coef_high[:, None])
        h_val = R * T * np.sum(coef * Temp, axis=0)
        return h_val


    def s(self, SpecieName, T, P):
        Tmin, Tmid, Tmax = self.__Species[SpecieName][ThermoData.TRANGE]
        coef_low = self.__Species[SpecieName][ThermoData.LOWTDATA]
        coef_high = self.__Species[SpecieName][ThermoData.HIGHTDATA]

        Temp = np.vstack([np.log(T), T, T**2 / 2, T**3 / 3, T**4 / 4, np.zeros_like(T), np.ones_like(T)])  # shape (7, len(T))
        coef = np.where(T <= Tmid, coef_low[:, None], coef_high[:, None])
        s_val = np.sum(R * coef * Temp, axis=0) - R * np.log(P/Pref)

        return s_val


    def g(self, SpecieName, T, P):
        return (self.h(SpecieName, T, P) - self.s(SpecieName, T, P) * T)

    def g0(self, SpecieName, T):
        return (self.h(SpecieName, T, Pref) - self.s(SpecieName, T, Pref) * T)

    def MM(self, SpecieName):
        return (self.__Species[SpecieName][ThermoData.MolarMass])

    def cp_mix(self, T, P, Y):
        ncells, nspecies = Y.shape
        cp_m = np.zeros(ncells)

        species_list = list(self.__Species.keys())
        for i, specie in enumerate(species_list):
            cp_i = self.cp(specie, T, P)
            cp_m += Y[:, i] * cp_i

        return cp_m

    def cv_mix(self, T, P, Y):
        return self.cp_mix(T, P, Y) - R

    def h_mix(self, T, P, Y):
        ncells, nspecies = Y.shape
        h_m = np.zeros(ncells)

        species_list = list(self.__Species.keys())
        for i, specie in enumerate(species_list):
            h_i = self.h(specie, T, P)
            h_m += Y[:, i] * h_i

        return h_m

    def s_mix(self, T, P, Y):
        ncells, nspecies = Y.shape
        s_m = np.zeros(ncells)

        species_list = list(self.__Species.keys())
        for i, specie in enumerate(species_list):
            s_i = self.s(specie, T, P)
            s_m += Y[:, i] * s_i

        return s_m

    def g_mix(self, T, P, Y):
        return self.h_mix(T, P, Y) - T * self.s_mix(T, P, Y)

    def g0_mix(self, T, Y):
        return self.g_mix(T, Pref, Y)

class ThermoData_vectorized:
    ##############
    ##### Retourne tout en massique
    #############
    def __init__(self, species_names, Tmin, Tmid, Tmax, coef_low, coef_high, molar_masses):
        self.species = species_names  # (nspecies,)
        self.Tmin = np.array(Tmin)  # (nspecies,)
        self.Tmid = np.array(Tmid)  # (nspecies,)
        self.Tmax = np.array(Tmax)  # (nspecies,)
        self.coef_low = np.array(coef_low)  # (nspecies, 6)
        self.coef_high = np.array(coef_high)  # (nspecies, 6)
        self.molar_masses = np.array(molar_masses)  # (nspecies,)
        self.inv_molar_masses = 1.0 / self.molar_masses
        self.nspecies = len(species_names)

    def _select_coef(self, T):
        """
        Sélectionne les bons coefficients (low ou high) en fonction de T.
        T: (ncells, 1)
        retourne coef: (ncells, nspecies, 7)
        """
        mask_low = T <= self.Tmid
        coef_low = self.coef_low[None, :, :]
        coef_high = self.coef_high[None, :, :]
        return np.where(mask_low[:, :, None], coef_low, coef_high)

    def cp_mix(self, T, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]  # (ncells, 1)
        coef = self._select_coef(T)[:, :, :5]  # (ncells, nspecies, 5)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = np.stack([np.ones_like(T), T, T_2, T_3, T_4], axis=-1)  # (ncells, 1, 5)
        cp_species_molar = R * np.sum(coef * Temp, axis=-1)  # (ncells, nspecies)
        cp_species_mass = cp_species_molar * self.inv_molar_masses
        return np.sum(Y * cp_species_mass, axis=1)  # (ncells,)

    def cv_mix(self, T, Y):
        #Y: (ncells, nspecies)
        return self.cp_mix(T, Y) - self.R_spec_from_Y(Y)  # relation Cp-Cv = Rspec

    def h_mix(self, T, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]
        coef = self._select_coef(T)[:, :, :6]  # (ncells, nspecies, 6)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = np.stack([np.ones_like(T), T / 2, T_2 / 3, T_3 / 4, T_4 / 5, 1 / T], axis=-1)  # (ncells, 1, 6)

        h_species_molar = R * T * np.sum(coef * Temp, axis=-1)  # (ncells, nspecies), J/mol
        h_species_mass = h_species_molar * self.inv_molar_masses  # J/kg

        return np.sum(Y * h_species_mass, axis=1)

    def s_mix(self, T, P, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]
        P = P[:, None]
        coef = self._select_coef(T)  # (ncells, nspecies, 7)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = np.stack([np.log(T), T, T_2 / 2, T_3 / 3, T_4 / 4, np.zeros_like(T), np.ones_like(T)], axis=-1)  # (ncells, 1, 7)

        s_species_molar = np.sum(R * coef * Temp, axis=-1) - R * np.log(P / Pref)  # J/mol/K
        s_species_mass = s_species_molar * self.inv_molar_masses  # J/kg/K

        return np.sum(Y * s_species_mass, axis=1)

    def g_mix(self, T, P, Y):
        #Y: (ncells, nspecies)
        return self.h_mix(T, Y) - T * self.s_mix(T, P, Y)

    def g0_mix(self, T, Y):
        #Y: (ncells, nspecies)
        return self.g_mix(T, Pref, Y)

    def R_spec_from_Y(self, Y):
        # returns R_specific for each cell (J/kg/K)
        Mmix = self.mean_molar_mass(Y)  # kg/mol
        return R / Mmix  # (N,)

    def mean_molar_mass(self, Y):
        # Y : (N, nspecies) mass fractions
        # Mmix = 1 / sum(Y_i / W_i)
        return 1.0 / np.sum(Y * self.inv_molar_masses[np.newaxis, :], axis=1)  # (N,)

if __name__ == '__main__':
    import time
    y = yamlParser(r'C:\Users\bromain\anaconda3\Lib\site-packages\cantera\data\gri30.yaml')
    th = ThermoData()
    y.populateThermoObject(th)
    vec_th = y.populateVectorizedThermoObject()
    #print(th.Species)
    cells = 100000
    x = np.arange(cells)
    T = 300.0 + 0.0 * x
    P = 101325.0 + 0.0 * x
    Y = np.zeros((cells, vec_th.nspecies))
    Y[:,0] = 1.0
    t0 = time.time()
    print(th.cp_mix(T, P, Y))
    print(time.time() - t0)
    t0 = time.time()
    print(vec_th.cp_mix(T, Y))
    print(time.time() - t0)

    print(vec_th.molar_masses)