import NeuralCFD.thermo.yaml_thermo as tp

import torch

class thermo:

    def __init__(self, thermoFilePath, dtype, device):

        self.dtype = dtype
        self.device = device

        self.thermoFilePath = thermoFilePath
        yamlFile = tp.yamlParser(self.thermoFilePath)
        self.thermoProp = yamlFile.populateVectorizedThermoObject(dtype=dtype, device=device)

    @staticmethod
    def torch_interp(x, xp, fp):
        """
        Interpolation linéaire 1D torch (équivalent à np.interp).
        x : scalaire ou tensor de valeurs à interpoler
        xp : tensor 1D (monotone croissant)
        fp : tensor 1D des valeurs correspondantes
        """
        # db_gen'assurer que xp est croissant
        indices = torch.searchsorted(xp, x.clamp(xp[0], xp[-1]))  # indices des intervalles
        indices = indices.clamp(1, len(xp) - 1)

        x0, x1 = xp[indices - 1], xp[indices]
        y0, y1 = fp[indices - 1], fp[indices]

        slope = (y1 - y0) / (x1 - x0)
        return(y0 + slope * (x - x0))

    def compute_etab(self, Ttab, Y):
        """
        Evaluer etab pour toutes les cellules d’un coup.
        Ttab: (M,)
        Y: (N, nspecies)
        vec_th: instance de ThermoData_vectorized

        Retour:
            etab: (N, M) avec etab[i,k] = h_mix(Ttab[k], Y[i]) - Rspec(Y[i])*Ttab[k]
        """
        N = Y.shape[0]
        M = Ttab.size(dim=0)

        # T répété pour chaque cellule
        T_flat = torch.tile(Ttab, (N,))  # (N*M,)
        # Y répété M fois par cellule
        Y_rep = torch.repeat_interleave(Y, M, dim=0)  # (N*M, nspecies)

        # h_mix vectorisé
        h_flat = self.thermoProp.h_mix(T_flat, Y_rep)  # (N*M,)
        h_mat = h_flat.reshape(N, M)

        Rspec = self.thermoProp.R_spec_from_Y(Y)[:, None]  # (N,1)
        etab = h_mat - Rspec * Ttab[None, :]  # (N,M)
        return(etab)

    def e_from_T(self, T, Y):
        # internal energy per mass e(T,Y) = h(T,Y) - R_spec * T
        h = self.thermoProp.h_mix(T, Y)  # (N,)
        Rspec = self.thermoProp.R_spec_from_Y(Y)  # (N,)
        return(h - Rspec * T)  # (N,)

    def expand_T_guess(self, T_prev):
        """
        Expand cell-centered T guess (N,) to interface T guess (N+1,)
        by arithmetic averaging.
        """
        N = T_prev.shape[0]
        T_int = torch.zeros(N + 1, device=self.device, dtype=self.dtype)
        # interfaces internes : moyenne des cellules voisines
        T_int[1:N] = 0.5 * (T_prev[:-1] + T_prev[1:])
        # bords : extrapolation simple
        T_int[0] = T_prev[0]
        T_int[-1] = T_prev[-1]
        return(T_int)

    def expand_T_guess_linear(self, T_prev):
        N = T_prev.shape[0]
        T_int = torch.zeros(N + 1, device=self.device, dtype=self.dtype)
        # extrapolation à gauche et droite
        T_int[0] = 2 * T_prev[0] - T_prev[1]
        T_int[-1] = 2 * T_prev[-1] - T_prev[-2]
        # moyennes internes
        T_int[1:N] = 0.5 * (T_prev[:-1] + T_prev[1:])
        return(T_int)

    def build_Ttab(self, Tmin=200.0, Tmax=6000.0, Ntab=300):
        """Créer la grille de températures pour la table (1D)."""
        return(torch.linspace(Tmin, Tmax, Ntab, device=self.device, dtype=self.dtype))

    def guess_T_from_table(self, e, Y, Tmin=200.0, Tmax=6000.0, Ntab=300):
        """
        Estimer T à partir de e et Y en interpolant sur une table.
        e: (N,)
        Y: (N, nspecies)
        vec_th: ThermoData_vectorized
        Retour: T_guess (N,)
        """
        N = e.shape[0]
        Ttab = self.build_Ttab(Tmin, Tmax, Ntab)  # (M,)
        etab = self.compute_etab(Ttab, Y)  # (N,M)

        T_guess = torch.empty(N, dtype=self.dtype, device=self.device)

        # interpolation ligne par ligne (un petit for sur N)
        for i in range(N):
            et = etab[i]
            ei = e[i]

            if et[0] <= et[-1]:
                T_guess[i] = self.torch_interp(ei, et, Ttab)
            else:
                T_guess[i] = self.torch_interp(ei, torch.flip(et, dims=[0]), torch.flip(Ttab, dims=[0]))

        return(T_guess)


    def solve_T_from_e_rho(self, e, Y, T_guess=None, maxiter=50, tol=1e-6, max_T_step=0.2, err_type='absolute'):
        Rspec = self.thermoProp.R_spec_from_Y(Y)
        #T_guess = None
        if T_guess is None:
            T = self.guess_T_from_table(e, Y)
        else:
            T = torch.clip(T_guess.clone(), 200.0, 6000.0)

        for i in range(maxiter):
            h = self.thermoProp.h_mix(T, Y)
            f = h - Rspec*T - e
            cp = self.thermoProp.cp_mix(T, Y)
            df = cp - Rspec
            df_safe = torch.where(torch.abs(df)<1e-12, torch.sign(df)*1e-12, df)
            delta = -f / df_safe
            max_step = max_T_step*T
            delta = torch.clip(delta, -max_step, max_step)
            Tnew = torch.clip(T + delta, 1.0, 2e4)

            if err_type == 'relative':
                err = torch.max(torch.abs(f) / (torch.abs(e)+1.0))
            elif err_type == 'absolute':
                err = torch.max(torch.abs(f))
            else:
                raise ValueError('unknown error computation type')
            if err < tol:
                return Tnew
            T = Tnew

        # fallback bisection si Newton rate
        #TODO: apply fallback only where it did not converged
        Tmin, Tmax = 1.0, 2e4
        N = e.shape[0]
        a = torch.full((N,), Tmin, device=self.device, dtype=self.dtype)
        b = torch.full((N,), Tmax, device=self.device, dtype=self.dtype)

        fa = self.thermoProp.h_mix(a, Y) - Rspec * a - e
        fb = self.thermoProp.h_mix(b, Y) - Rspec * b - e

        # Si fa et fb ont le même signe → pas de racine trouvée
        mask_fail = (fa * fb > 0)

        mid = torch.empty_like(e)
        for i in range(maxiter):
            mid = 0.5 * (a + b)
            fm = self.thermoProp.h_mix(mid, Y) - Rspec * mid - e

            # critère de convergence
            if torch.all(torch.abs(fm) < tol):
                break

            # met à jour bornes en vectoriel
            left_mask = fa * fm < 0
            right_mask = ~left_mask

            b = torch.where(left_mask, mid, b)
            fb = torch.where(left_mask, fm, fb)

            a = torch.where(right_mask, mid, a)
            fa = torch.where(right_mask, fm, fa)

        # résultat
        T_out = mid
        # pour les cas sans racine, on renvoie Tmin par défaut (ou un guess)
        T_out = torch.where(mask_fail, torch.full_like(e, Tmin), T_out)

        if i == maxiter - 1:
            bad_idx = torch.where(torch.abs(f) > tol)[0]
            if bad_idx.size(dim=0) > 0:
                bi = bad_idx[:10]
                print("  example bad idx:", bi)
                print("  e[bi]:", e[bi])
                print("  Rspec[bi]:", Rspec[bi])
                print("  T_guess[bi]:", T[bi])
                print("  cp[bi]:", cp[bi])
                print("  df[bi] (cp-Rspec):", df[bi])
            raise RuntimeError('Maximum number of iterations reached for temperature computation')

        return(T_out)