import torch
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

    def populateVectorizedThermoObject(self, dtype, device):
        species_names = [v['name'] for v in self.thermo]
        Tmin = [v['trange'][0] for v in self.thermo]
        Tmid = [v['trange'][1] for v in self.thermo]
        Tmax = [v['trange'][2] for v in self.thermo]
        coefs_low = [v['data'][0] for v in self.thermo]
        coefs_high = [v['data'][1] for v in self.thermo]
        molar_masses = [sum([number * getattr(MM, atom) for number, atom in v['composition']]) for v in self.thermo]
        return(ThermoData_vectorized(species_names, Tmin, Tmid, Tmax, coefs_low, coefs_high, molar_masses, dtype, device))


class ThermoData_vectorized:
    ##############
    ##### Retourne tout en massique
    #############
    def __init__(self, species_names, Tmin, Tmid, Tmax, coef_low, coef_high, molar_masses, dtype, device):
        self.species = species_names  # (nspecies,)
        self.Tmin = torch.tensor(Tmin, dtype=dtype, device=device)  # (nspecies,)
        self.Tmid = torch.tensor(Tmid, dtype=dtype, device=device)  # (nspecies,)
        self.Tmax = torch.tensor(Tmax, dtype=dtype, device=device)  # (nspecies,)
        self.coef_low = torch.tensor(coef_low, dtype=dtype, device=device)  # (nspecies, 6)
        self.coef_high = torch.tensor(coef_high, dtype=dtype, device=device)  # (nspecies, 6)
        self.molar_masses = torch.tensor(molar_masses, dtype=dtype, device=device)  # (nspecies,)
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
        return torch.where(mask_low[:, :, None], coef_low, coef_high)

    def cp_mix(self, T, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]  # (ncells, 1)
        coef = self._select_coef(T)[:, :, :5]  # (ncells, nspecies, 5)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.ones_like(T), T, T_2, T_3, T_4], dim=-1)  # (ncells, 1, 5)
        cp_species_molar = R * torch.sum(coef * Temp, dim=-1)  # (ncells, nspecies)
        cp_species_mass = cp_species_molar * self.inv_molar_masses
        return torch.sum(Y * cp_species_mass, dim=1)  # (ncells,)

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
        Temp = torch.stack([torch.ones_like(T), T / 2, T_2 / 3, T_3 / 4, T_4 / 5, 1 / T], dim=-1)  # (ncells, 1, 6)

        h_species_molar = R * T * torch.sum(coef * Temp, dim=-1)  # (ncells, nspecies), J/mol
        h_species_mass = h_species_molar * self.inv_molar_masses  # J/kg

        return torch.sum(Y * h_species_mass, dim=1)

    def s_mix(self, T, P, Y):
        #Y: (ncells, nspecies)
        T = T[:, None]
        P = P[:, None]
        coef = self._select_coef(T)  # (ncells, nspecies, 7)
        T_2 = T * T
        T_3 = T_2 * T
        T_4 = T_3 * T
        Temp = torch.stack([torch.log(T), T, T_2 / 2, T_3 / 3, T_4 / 4, torch.zeros_like(T), torch.ones_like(T)], dim=-1)  # (ncells, 1, 7)

        s_species_molar = torch.sum(R * coef * Temp, dim=-1) - R * torch.log(P / Pref)  # J/mol/K
        s_species_mass = s_species_molar * self.inv_molar_masses  # J/kg/K

        return torch.sum(Y * s_species_mass, dim=1)

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
        return 1.0 / torch.sum(Y * self.inv_molar_masses[None, :], dim=1)  # (N,)

if __name__ == '__main__':
    import time
    dtype = torch.float64
    device = 'cpu'
    y = yamlParser(r'gri30.yaml')
    vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
    #print(th.Species)
    cells = 100000
    x = torch.arange(cells, dtype=dtype, device=device)
    T = 300.0 + 0.0 * x
    P = 101325.0 + 0.0 * x
    Y = torch.zeros((cells, vec_th.nspecies), dtype=dtype, device=device)
    Y[:,0] = 1.0
    t0 = time.time()
    print(vec_th.cp_mix(T, Y))
    print(time.time() - t0)

    print(vec_th.molar_masses)