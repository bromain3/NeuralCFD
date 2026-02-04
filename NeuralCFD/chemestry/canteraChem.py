import torch
import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from NeuralCFD.chemestry.DummyChemistry import DummyChemistry

class canteraChem(DummyChemistry):

    def __init__(self, mech, ncells):
        self.mech = mech
        self.gas = ct.Solution(mech)  # un seul gaz de référence
        self.ncells = ncells

    def computeIntegratedTPY(self, grid, dt, model='cst_rho'):
        nspecies = grid.nspecies
        ncells = grid.ncells

        # --- Extraction des champs ---
        T_np   = grid.cells.T.cpu().numpy().copy()
        Y_np   = grid.cells.Ys.cpu().numpy().copy()
        rho_np = grid.cells.rho.cpu().numpy().copy()

        T_new = torch.zeros_like(grid.cells.p)
        P_new = torch.zeros_like(grid.cells.p)
        Ys_new_numpy = np.zeros_like(grid.cells.Ys.cpu())


        # --- Boucle sur les cellules ---
        if model == 'cst_rho':
            # TODO: directly extract Q where concentration is stored instaed of using Ys and then multiplying
            for i in range(ncells):
                # État initial de Cantera
                self.gas.TDY = T_np[i], rho_np[i], Y_np[i]
                y0 = np.empty(nspecies + 1)
                y0[1:] = Y_np[i]
                y0[0] = T_np[i]
                #nsp = len(self.gas.molecular_weights)
                #J = np.zeros((1 + nsp, 1 + nsp))
                def rhs(t, y):
                    """
                    y[0] = T
                    y[1:1+nsp] = mass fractions Y_k (length nsp)
                    returns dy/dt same shape
                    """
                    T = y[0]
                    Y = y[1:]
                    # densité fixée (masse totale m_total, volume fixé V)
                    rho = rho_np[i]  # kg/m^3 (constante dans ce cas fermé)
                    # imposer l'état sur l'objet Cantera par (T, rho, Y)
                    # setState_TRY / setState_TRY exists sous l'API comme setState_TRY
                    # méthode haut-niveau: setState_TRY(t, rho, mass_fractions)
                    self.gas.TDY = T, rho, Y
                    # net production rates (kmol / m^3 / db_gen)
                    omega_dot = self.gas.net_production_rates  # array length nsp, kmol/m^3/db_gen
                    # molecular weights [kg/kmol]
                    W = self.gas.molecular_weights  # array length nsp, kg/kmol
                    # partial molar internal energies [J/kmol]
                    ubar = self.gas.partial_molar_int_energies  # J / kmol
                    # convert partial molar internal energies to specific (J/kg):
                    # u_k [J/kg] = ubar[k] [J/kmol] / W[k] [kg/kmol]
                    # (on suppose W[k] > 0)
                    u_k = ubar / W  # J/kg
                    # mixture cv (J/kg/K)
                    cv = self.gas.cv_mass
                    # dY/dt = omega_dot * W / rho  (omega_dot in kmol/m3/db_gen, W in kg/kmol -> kg/m3/db_gen divided by rho kg/m3 -> 1/db_gen)
                    dYdt = (omega_dot * W) / rho
                    # dT/dt = -(1/(rho * cv)) * sum_k (omega_dot * W * u_k)
                    # note: omega_dot * W * u_k -> (kg/m^3/db_gen) * (J/kg) = J/m^3/db_gen
                    sum_term = np.dot(omega_dot * W, u_k)  # scalar J/m^3/db_gen
                    dTdt = - (1.0 / (rho * cv)) * sum_term
                    # construire le vecteur dérivée
                    dydt = np.empty(1 + nspecies)
                    dydt[0] = dTdt
                    dydt[1:] = dYdt
                    return dydt

                # Intégration chimique sur [0, dt]
                sol = solve_ivp( #TODO: compute analytical jacobian to improve computation time
                    rhs,
                    (0.0, dt),
                    y0,
                    method='BDF',
                    #atol=1e-12,
                    #rtol=1e-8,
                    vectorized=False,
                    #jac=jacobian,
                )

                if not sol.success:
                    print(f"[WARN] Integration failed in cell {i}: {sol.message}")
                    y_end = y0
                else:
                    y_end = sol.y[:, -1]

                T_new[i] = y_end[0]
                Ys_new_numpy[i] = y_end[1:]


                self.gas.TDY = T_new[i], rho_np[i], y_end[1:]
                P_new[i] = self.gas.P

        elif model == 'cst_p': #TODO: validate this model because it seems to be wrong
            # --- Modèle à pression constante ---
            # On considère la pression initiale du gaz dans chaque cellule comme la pression constante.
            # Pour chaque cellule on intègre T et Y sur [0, dt] en imposant p constant via gas.TPY = T, p, Y.
            for i in range(ncells):
                # Obtenir pression initiale dans la cellule (depuis l'état initial T, rho, Y)
                # On calcule p via Cantera en fixant l'état initial (T, rho, Y)
                self.gas.TDY = T_np[i], rho_np[i], Y_np[i]
                p_cell = self.gas.P  # pression à garder constante (Pa)
                y0 = np.empty(nspecies + 1)
                y0[1:] = Y_np[i]
                y0[0] = T_np[i]
                def rhs(t, y):
                    """
                    y[0] = T
                    y[1:] = mass fractions Y_k (length nsp)
                    returns dy/dt same shape, with pressure p_cell kept constant.
                    """
                    T = y[0]
                    Y = y[1:]
                    # imposer l'état sur l'objet Cantera par (T, p, Y)
                    # cela permet à Cantera de fournir la densité correcte pour p constant
                    self.gas.TPY = T, p_cell, Y
                    # densité actuelle (kg/m^3) — variable dans ce modèle
                    rho = self.gas.density  # kg/m^3
                    # net production rates (kmol / m^3 / db_gen)
                    omega_dot = self.gas.net_production_rates  # array length nsp, kmol/m^3/db_gen
                    # molecular weights [kg/kmol]
                    W = self.gas.molecular_weights  # array length nsp, kg/kmol
                    # partial molar enthalpies [J/kmol]
                    hbar = self.gas.partial_molar_enthalpies  # J / kmol
                    # mixture cp (J/kg/K)
                    cp = self.gas.cp_mass
                    # dY/dt = omega_dot * W / rho  (same forme mais rho variable)
                    dYdt = (omega_dot * W) / rho
                    # sum_term for enthalpy: omega_dot [kmol/m3/db_gen] * hbar [J/kmol] -> J/m3/db_gen
                    sum_term = np.dot(omega_dot, hbar)  # scalar J/m^3/db_gen
                    # dT/dt = -(1 / (rho * cp)) * sum_term
                    dTdt = - (1.0 / (rho * cp)) * sum_term
                    # construire le vecteur dérivée
                    dydt = np.empty(1 + nspecies)
                    dydt[0] = dTdt
                    dydt[1:] = dYdt
                    return dydt

                # Intégration chimique sur [0, dt]
                sol = solve_ivp(
                    rhs,
                    (0.0, dt),
                    y0,
                    method='BDF',
                    vectorized=False,
                )

                if not sol.success:
                    print(f"[WARN] Integration failed in cell {i}: {sol.message}")
                    y_end = y0
                else:
                    y_end = sol.y[:, -1]

                T_new[i] = y_end[0]
                Ys_new_numpy[i] = y_end[1:]
                P_new[i] = p_cell

        else:
            raise NotImplementedError('The only available models are "cst_rho" and "cst_p"')

        return T_new, P_new, torch.from_numpy(Ys_new_numpy).to(device=T_new.device, dtype=T_new.dtype)
