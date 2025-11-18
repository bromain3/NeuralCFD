import torch

import NeuralCFD.grid.state as st
from NeuralCFD.interpolation.firstOrderInterp import FirstOrderInterpolation

class grid:

    def __init__(self, L, ncells, thermo, device, dtype, interpolationMethode=FirstOrderInterpolation(), custome_S=1.0, custome_dx=None):
        #TODO: implement a varying mesh that will affect the transport solver
        assert type(ncells) == int
        assert type(L) == float

        self.L = L
        self.ncells = ncells

        self.nspecies = thermo.thermoProp.nspecies

        self.thermo = thermo

        self.device = device
        self.dtype = dtype

        self.interpolationMethode = interpolationMethode

        # Creates everything with ghost cells
        self.__createSpatialProperties(custome_S, custome_dx)
        self.__createQ()
        self.__createInterfaces()

    def updateCellsState(self, Q, bc, usePrevT=True, interpolationMethode='1st_order', interfaceThermoInterp=False):
        #TODO: implement interpolationMethode and interfaceTherom interp
        # assigne new Q (and internal recompute of thermo props)
        self.cells.updateQ(Q, self.thermo, usePrevT=usePrevT)

        # update interior interfaces (and internal recompute of thermo props OR extrapolation for computation efficiency)
        #TODO: implement possibility to interpolate natural quantities instead of recomputing theme
        self.interpolationMethode.interpolate(self.leftInterfaces, self.rightInterfaces, self.cells)

        self.applyBoundaryConditions(bc)
        self.leftInterfaces.updateThermoFromQ(self.thermo, usePrevT=usePrevT) #TODO: maybe differenciate classes for interfaces and cell centers to store exactly the usefull values. and interpolate also thermo props even if it means small inconstiencies to prevent from calling too much the thermo solver
        self.rightInterfaces.updateThermoFromQ(self.thermo, usePrevT=usePrevT)

    def applyBoundaryConditions(self, bc):
        #TODO: ask for a dict with left boundary and right boundary and treat independently, or ask for bc classes to handle bc on each side
        if bc['type'] == 'periodic':
            # interface 0 : between cell N-1 (left) and cell 0 (right)
            self.leftInterfaces.Q[:, 0] = self.cells.Q[:, -1]
            # interface N : between cell N-1 (left) and cell 0 (right) as well (periodic wrap)
            # here we set the ghost on the right of the last cell equal to the first cell (periodic)
            self.rightInterfaces.Q[:, -1] = self.cells.Q[:, 0]
        elif bc['type'] == 'flame1d_P_out_eq_P_in':
            if 'inflow_state' not in bc.keys():
                raise ValueError("Need inflow_state for bc='flame1d_P_out_eq_P_in'")
            inflow_state = bc['inflow_state']

            # left boundary (inflow) : ghost cell fixed to inflow state
            # left boundary (interface 0): left=ghost(inflow), right=cell 0
            self.leftInterfaces.Q[:, 0] = inflow_state

            # right boundary (outflow) : ghost cell = last interior cell (zero-gradient)
            # right boundary (interface N): left = last cell (N-1), right = ghost = copy of last cell => zero gradient
            self.rightInterfaces.Q[:, -1] = self.cells.Q[:, -1]
        elif bc['type'] == 'flame1d':
            if 'inflow_state' not in bc.keys():
                raise ValueError("Need inflow_state for bc='flame1d'")
            elif 'P_out' not in bc.keys():
                raise ValueError("Need P_out for bc='flame1d'")
            inflow_state = bc['inflow_state']
            P_out = bc['P_out']

            # left boundary (inflow) : ghost cell fixed to inflow state
            # left boundary (interface 0): left=ghost(inflow), right=cell 0
            self.leftInterfaces.Q[:, 0] = inflow_state

            # right boundary (outflow) : ghost cell = last interior cell (zero-gradient)
            # right boundary (interface N): left = last cell (N-1), right = ghost = copy of last cell => zero gradient
            Q_out = torch.zeros_like(self.cells.Q[:, -1])
            Rspec = self.thermo.thermoProp.R_spec_from_Y(self.cells.Ys[-1])
            rho = P_out / Rspec / self.cells.T[-1]
            Q_out[0] = rho
            Q_out[1] = rho * self.cells.u[-1]
            Q_out[2] = rho * self.cells.Q[2, -1] / self.cells.Q[0, -1]
            Q_out[3:3 + self.nspecies] = rho * self.cells.Q[3:3 + self.nspecies, -1] / self.cells.Q[0, -1]
            self.rightInterfaces.Q[:, -1] = Q_out
        elif bc['type'] == 'closed':
            # Left boundary (mur à gauche)
            self.leftInterfaces.Q[:, 0] = self.cells.Q[:, 0]
            # inversion de la vitesse (rho*u)
            self.leftInterfaces.Q[1, 0] *= -1
            #self.rightInterfaces.Q[1, 0] *= -1

            # Right boundary (mur à droite)
            self.rightInterfaces.Q[:, -1] = self.cells.Q[:, -1]
            # inversion de la vitesse (rho*u)
            #self.leftInterfaces.Q[1, -1] *= -1
            self.rightInterfaces.Q[1, -1] *= -1
        elif bc['type'] == 'outlets_P_in_eq_P_out':
            # left boundary (outflow) : ghost cell fixed to inflow state
            # left boundary (interface 0): left=ghost(inflow), right=cell 0
            self.leftInterfaces.Q[:, 0] = self.cells.Q[:, 0]

            # right boundary (outflow) : ghost cell = last interior cell (zero-gradient)
            # right boundary (interface N): left = last cell (N-1), right = ghost = copy of last cell => zero gradient
            self.rightInterfaces.Q[:, -1] = self.cells.Q[:, -1]
        elif bc['type'] == 'symetric':
            # copie miroir : rho et énergie égaux, vitesse inversée
            Q_sym = torch.clone(self.cells.Q[:, 0])
            Q_sym[1] = -Q_sym[1]  # inversion de la vitesse
            self.leftInterfaces.Q[:, 0] = Q_sym
            # right boundary (outflow) : ghost cell = last interior cell (zero-gradient)
            # right boundary (interface N): left = last cell (N-1), right = ghost = copy of last cell => zero gradient
            self.rightInterfaces.Q[:, -1] = self.cells.Q[:, -1]
        elif bc['type'] == 'outlets':
            if 'P_out' not in bc.keys():
                raise ValueError("Need P_out for bc='outlets'")
            P_out = bc['P_out']

            Q_out_right = torch.zeros_like(self.cells.Q[:, -1])
            Rspec = self.thermo.thermoProp.R_spec_from_Y(self.cells.Ys[-1])
            rho = P_out / Rspec / self.cells.T[-1]
            Q_out_right[0] = rho
            Q_out_right[1] = rho * self.cells.u[-1]
            Q_out_right[2] = rho * self.cells.Q[2, -1] / self.cells.Q[0, -1]
            Q_out_right[3:3 + self.nspecies] = rho * self.cells.Q[3:3 + self.nspecies, -1] / self.cells.Q[0, -1]
            self.rightInterfaces.Q[:, -1] = Q_out_right

            Q_out_left = torch.zeros_like(self.cells.Q[:, 0])
            Rspec = self.thermo.thermoProp.R_spec_from_Y(self.cells.Ys[0])
            rho = P_out / Rspec / self.cells.T[0]
            Q_out_left[0] = rho
            Q_out_left[1] = rho * self.cells.u[0]
            Q_out_left[2] = rho * self.cells.Q[2, 0] / self.cells.Q[0, 0]
            Q_out_left[3:3 + self.nspecies] = rho * self.cells.Q[3:3 + self.nspecies, 0] / self.cells.Q[0, 0]
            self.leftInterfaces.Q[:, 0] = Q_out_left
        elif bc['type'] == 'outlets_nonreflective':
            #TODO: add regime detection methode to change computation if M > 1
            if 'P_out' not in bc.keys():
                raise ValueError("Need P_out for bc='outlets_nonreflective'")
            P_out = bc['P_out']
            if 'beta' not in bc.keys():
                raise ValueError("Need beta (relaxation coef) for bc='outlets_nonreflective'")
            beta = bc['beta']
            print('WARNING, this boundary condition is in dev and not working for the moment')
            ns = self.nspecies

            #beta = 0.2–0.5 donne souvent le meilleur compromis entre stabilité et faible réflexion.
            #Si tu observes des oscillations de pression au bord: réduis beta.
            #Si la pression met trop de temps à atteindre la cible: augmente beta
            # beta ≈ 0.2–0.5 : bon compromis pour limiter les réflexions.
            # Tu peux aussi faire varier beta selon Mach
            #beta = 0.3 * torch.exp(-abs(u_in) / a_in)
            #plus d’absorption quand le flux est lent (ondes faibles).
            #β=0 → condition non-réfléchissante pure,
            #β=1 → pression imposée strictement.

            # ======================================================
            # === Sortie droite (x = L) ============================
            # ======================================================
            u_in = self.cells.u[-1]
            T_in = self.cells.T[-1]
            Ys_in = self.cells.Ys[-1]
            P_in = self.cells.p[-1]
            gamma_in = self.cells.gamma[-1]
            Rspec = self.thermo.thermoProp.R_spec_from_Y(Ys_in)
            a_in = torch.sqrt(gamma_in * Rspec * T_in)
            rho_in = self.cells.Q[0, -1]

            # Invariants
            W_plus = u_in + 2 * a_in / (gamma_in - 1)
            W_minus = u_in - 2 * a_in / (gamma_in - 1)

            # --- Condition non-réfléchissante avec pression cible ---
            p_target = P_out

            # onde entrante (gauche → droite) : W_minus
            W_minus_boundary = W_minus + beta * (p_target - P_in) / (rho_in * a_in)

            # Reconstruction
            u_out = 0.5 * (W_plus + W_minus_boundary)
            a_out = 0.25 * (gamma_in - 1) * (W_plus - W_minus_boundary)
            T_out = a_out ** 2 / (gamma_in * Rspec)
            P_out = p_target
            rho_out = P_out / (Rspec * T_out)

            # Conservatif
            Q_out = torch.zeros_like(self.cells.Q[:, -1])
            Q_out[0] = rho_out
            Q_out[1] = rho_out * u_out
            Q_out[2] = P_out / (gamma_in - 1) + 0.5 * rho_out * u_out ** 2
            Q_out[3:3 + ns] = rho_out * Ys_in

            self.rightInterfaces.Q[:, -1] = Q_out

            # ======================================================
            # === Sortie gauche (x = 0) ============================
            # ======================================================
            u_in_l = self.cells.u[0]
            T_in_l = self.cells.T[0]
            Ys_in_l = self.cells.Ys[0]
            P_in_l = self.cells.p[0]
            gamma_l = self.cells.gamma[0]
            Rspec_l = self.thermo.thermoProp.R_spec_from_Y(Ys_in_l)
            a_in_l = torch.sqrt(gamma_l * Rspec_l * T_in_l)
            rho_in_l = self.cells.Q[0, 0]

            # Invariants
            W_plus_l = u_in_l + 2 * a_in_l / (gamma_l - 1)
            W_minus_l = u_in_l - 2 * a_in_l / (gamma_l - 1)

            # --- Condition non-réfléchissante avec pression cible ---
            p_target_left = P_out

            # onde entrante (droite → gauche) : W_plus
            W_plus_boundary = W_plus_l + beta * (p_target_left - P_in_l) / (rho_in_l * a_in_l)

            # Reconstruction
            u_out_l = 0.5 * (W_plus_boundary + W_minus_l)
            a_out_l = 0.25 * (gamma_l - 1) * (W_plus_boundary - W_minus_l)
            T_out_l = a_out_l ** 2 / (gamma_l * Rspec_l)
            rho_out_l = P_in_l / (Rspec_l * T_out_l)
            P_out_l = rho_out_l * Rspec_l * T_out_l

            # Conservatif
            Q_out_l = torch.zeros_like(self.cells.Q[:, 0])
            Q_out_l[0] = rho_out_l
            Q_out_l[1] = rho_out_l * u_out_l
            Q_out_l[2] = P_out_l / (gamma_l - 1) + 0.5 * rho_out_l * u_out_l ** 2
            Q_out_l[3:3 + ns] = rho_out_l * Ys_in_l

            self.leftInterfaces.Q[:, 0] = Q_out_l


        else:
            raise NotImplementedError("Only 'periodic', 'flame1d', 'flame1d_P_out_eq_P_in', 'closed', 'outlets', 'outlets_nonreflective' BC implemented")


    def __createSpatialProperties(self, custome_S=1.0, custome_dx=None):
        if custome_dx is not None:
            print('WARNING: Using a custom_dx is not explicitely physicaly validated')
            if isinstance(custome_dx, (int, float)):
                raise ValueError(f'Custom dx must be a torch.Tensor')
            else:
                if custome_dx.dim() != 1:
                    raise ValueError(f'Custom dx must be a 1D tensor but it has {custome_dx.dim()} dimensions')
                if len(custome_dx) != self.ncells:
                    raise ValueError(f'Custom dx must be a 1D tensor of length {self.ncells} but has length {len(custome_dx)}')
                self.dx = custome_dx.to(device=self.device, dtype=self.dtype)
        else:
            self.dx = self.L / self.ncells + torch.zeros(self.ncells, device=self.device, dtype=self.dtype)

        self.interfaces = torch.cat((torch.zeros(1, dtype=self.dx.dtype, device=self.dx.device), torch.cumsum(self.dx, dim=0))) #TODO: if we want to save storage, we could recompute that value in ALE requiring it
        self.x = self.interfaces[:-1] + self.dx / 2

        if custome_S is not None:
            print('WARNING: Using a custom_S is not explicitely physicaly validated')
            if isinstance(custome_S, (int, float)):
                self.S = custome_S + torch.zeros(self.ncells + 1, device=self.device, dtype=self.dtype)
            else:
                if custome_S.dim() != 1:
                    raise ValueError(f'Custom Surfaces must be a 1D tensor or a float/int but it has {custome_S.dim()} dimensions')
                if len(custome_S) != self.ncells + 1:
                    raise ValueError(f'Custom Surfaces must be a float/integer or must be a 1D tensor of length {self.ncells+1} but has length {len(custome_S)}')
                self.S = custome_S.to(device=self.device, dtype=self.dtype)

        self.V = 0.5 * (self.S[1:] + self.S[:-1]) * self.dx

    def __createQ(self):
        self.cells = st.state(self.ncells, self.nspecies, self.device, self.dtype)

    def __createInterfaces(self):
        self.leftInterfaces = st.state(self.ncells + 1, self.nspecies, self.device, self.dtype)
        self.rightInterfaces = st.state(self.ncells + 1, self.nspecies, self.device, self.dtype)


if __name__ == '__main__':
    g = grid(1.0, 100, 2, 'cpu', torch.float64)
    print(g.cells.Q)