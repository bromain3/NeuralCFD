import torch

class state:
    def __init__(self, ncells, nspecies, device, dtype):
        self.ncells = ncells
        self.nspecies = nspecies

        self.device = device
        self.dtype = dtype

        self.Q = torch.empty((3 + self.nspecies, self.ncells), device=self.device, dtype=self.dtype)

        self.T = torch.empty(self.ncells, device=self.device, dtype=self.dtype)
        self.cv  = torch.empty(self.ncells, device=self.device, dtype=self.dtype)
        self.gamma = torch.empty(self.ncells, device=self.device, dtype=self.dtype)
        self.a = torch.empty(self.ncells, device=self.device, dtype=self.dtype)
        self.rho = torch.empty(self.ncells, device=self.device, dtype=self.dtype)
        self.u = torch.empty(self.ncells, device=self.device, dtype=self.dtype)
        self.p = torch.empty(self.ncells, device=self.device, dtype=self.dtype)
        self.Ys = torch.empty((self.ncells, self.nspecies), device=self.device, dtype=self.dtype)  #TODO: Ys, u and e could be recomputed at every call to gain in VRAM at expense of little computation time. it will also help to implement function for updating thermo variables less often
        self.e = torch.empty(self.ncells, device=self.device, dtype=self.dtype)

    def updateThermoFromQ(self, thermo, usePrevT=True): #TODO: make a unified thermo solver that computes everything that we need and unly read the values in this function
        self.rho = self.Q[0, :]
        inv_rho = 1.0 / self.rho
        self.u = self.Q[1, :] * inv_rho
        self.e = (self.Q[2, :] * inv_rho) - 0.5 * self.u ** 2
        self.Ys = (self.Q[3: 3 + self.nspecies, :] * inv_rho).T
        if usePrevT:
            self.T = thermo.solve_T_from_e_rho(self.e, self.Ys, T_guess=self.T)
        else:
            self.T = thermo.solve_T_from_e_rho(self.e, self.Ys)
        Rspec = thermo.thermoProp.R_spec_from_Y(self.Ys)
        self.p = self.rho * Rspec * self.T
        self.cv = thermo.thermoProp.cv_mix(self.T, self.Ys)
        self.gamma = thermo.thermoProp.cp_mix(self.T, self.Ys) / self.cv
        self.a = torch.sqrt(torch.clamp(self.gamma * Rspec * self.T, min=0.0))

    def updateQ(self, Q, thermo, usePrevT=True):
        # change Q
        self.Q = Q

        # recompute thermo props
        self.updateThermoFromQ(thermo, usePrevT=usePrevT)