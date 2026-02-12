from abc import ABC, abstractmethod
import torch

class DummyCoagulation(ABC):
    def __init__(self, grid, RHO_SOOT):


        self.kB = torch.tensor(1.380649e-23, device=grid.device, dtype=grid.dtype)  # J/K
        self.RHO_SOOT = torch.tensor(RHO_SOOT, device=grid.device, dtype=grid.dtype)
        self.lambda_g = torch.tensor(68e-9, device=grid.device, dtype=grid.dtype)  # m (libre parcours moyen gaz)
        self.mu = torch.tensor(1.8e-5, device=grid.device, dtype=grid.dtype)  # Pa·s (viscosité dynamique)

        # Masse d'une particule dans une section donnée (kg)
        self.m_class = grid.meanParticuleVolume * self.RHO_SOOT

        self.coag_map = self.__build_conservative_coag_map(grid)

        self.__precompute_scatter_indices(grid.ncells, grid.nclasses, self.coag_map, grid)

        # Précalcul du masque triangulaire
        nclasses = grid.nclasses
        triu_mask = torch.triu(torch.ones(nclasses, nclasses, dtype=grid.dtype, device=grid.device))
        diag_mask = torch.eye(nclasses, dtype=grid.dtype, device=grid.device)
        self._triu_mask_coag = triu_mask - 0.5 * diag_mask

    @abstractmethod
    def compute_kernel_matrix(self, ns, ys, rhos, Ts):
        """
        :return: coagulation kernel (ncells, nclasses, nclasses)
        """
        raise NotImplementedError('Any coagulation class inheriting from DummyCoagulation must implement a compute_kernel_matrix method')


    def __build_conservative_coag_map(self, grid):
        """
        Construction de la carte de coagulation - VERSION COHÉRENTE avec basic_loops

        LOGIQUE (identique à _coagulation_basic_loops):
        Pour chaque paire (j, k):
        1. v_new = v_j + v_k
        2. Trouver k_low tel que: v[k_low] <= v_new < v[k_low+1]
        3. k_high = k_low + 1 (sauf si overflow → k_high = k_low)
        4. frac_low = (v_high - v_new) / (v_high - v_low)

        Cette carte sera utilisée par _coagulation_vectorized
        """
        nclasses = grid.nclasses
        v = self.m_class / self.RHO_SOOT  # (nclasses,) volumes en m³

        # Matrice des volumes résultants v_new[j,k] = v[j] + v[k]
        v_j = v[:, None]  # (nclasses, 1)
        v_k = v[None, :]  # (1, nclasses)
        v_new = v_j + v_k  # (nclasses, nclasses)

        # Initialisation des tenseurs de sortie
        k_low = torch.zeros((nclasses, nclasses), dtype=torch.int64, device=grid.device)
        k_high = torch.zeros((nclasses, nclasses), dtype=torch.int64, device=grid.device)
        frac_low = torch.zeros((nclasses, nclasses), dtype=grid.dtype, device=grid.device)

        # Remplissage - même algorithme que dans basic_loops
        for j in range(nclasses):
            for k in range(nclasses):
                v_target = v_new[j, k].item()

                # Chercher k_target_low tel que v[k_target_low] <= v_target < v[k_target_low+1]
                k_target_low = nclasses - 1  # Valeur par défaut (overflow)

                for i in range(nclasses - 1):
                    if v[i] <= v_target < v[i + 1]:
                        k_target_low = i
                        break

                k_target_high = min(k_target_low + 1, nclasses - 1)

                # Calcul de la fraction (interpolation linéaire)
                if k_target_low == k_target_high:
                    # Une seule classe de destination
                    frac = 1.0
                else:
                    # Deux classes: distribution proportionnelle
                    v_low = v[k_target_low].item()
                    v_high = v[k_target_high].item()

                    #if abs(v_high - v_low) > 1e-30:
                    frac = (v_high - v_target) / (v_high - v_low)
                    frac = max(0.0, min(1.0, frac))
                    #else:
                    #    frac = 1.0

                # Stockage
                k_low[j, k] = k_target_low
                k_high[j, k] = k_target_high
                frac_low[j, k] = frac

        # Validation
        assert torch.all(k_low >= 0) and torch.all(k_low < nclasses), "k_low hors limites"
        assert torch.all(k_high >= 0) and torch.all(k_high < nclasses), "k_high hors limites"
        assert torch.all(frac_low >= -1e-6), f"frac_low < 0: min={frac_low.min()}"
        assert torch.all(frac_low <= 1.0 + 1e-6), f"frac_low > 1: max={frac_low.max()}"

        return {
            'k_low': k_low,
            'k_high': k_high,
            'frac_low': torch.clamp(frac_low, 0.0, 1.0),
            'v_new': v_new
        }

    def __precompute_scatter_indices(self, ncells, nclasses, coag_map, grid):
        """Précalcul des indices pour scatter_add (version ultra)"""
        k_low = coag_map['k_low']
        k_high = coag_map['k_high']
        frac_low = coag_map['frac_low']

        cell_idx = torch.arange(ncells, device=grid.device, dtype=torch.int64)[:, None, None]
        cell_idx = cell_idx.expand(ncells, nclasses, nclasses)

        k_low_exp = k_low[None, :, :].expand(ncells, nclasses, nclasses)
        k_high_exp = k_high[None, :, :].expand(ncells, nclasses, nclasses)

        flat_idx_low = (cell_idx * nclasses + k_low_exp).flatten()
        flat_idx_high = (cell_idx * nclasses + k_high_exp).flatten()

        frac_low_exp = frac_low[None, :, :].expand(ncells, nclasses, nclasses)
        frac_low_flat = frac_low_exp.flatten()

        self._flat_idx_low = flat_idx_low
        self._flat_idx_high = flat_idx_high
        self._frac_low_flat = frac_low_flat

    def dndt(self, ns, ys, rhos, Ts):
        """
        Version ULTRA optimisée – conservative
        Aucun for-loop, aucun recalcul d'indices
        """
        K = self.compute_kernel_matrix(ns, ys, rhos, Ts)

        # =========================
        # COLLISION UNIQUE j <= k
        # =========================
        n_j = ns[:, :, None]
        n_k = ns[:, None, :]

        collision_rate = K * n_j * n_k
        collision_rate = collision_rate * self._triu_mask_coag[None, :, :]

        # =========================
        # PERTE
        # =========================
        loss_j = torch.sum(collision_rate, dim=2)
        loss_k = torch.sum(collision_rate, dim=1)

        dndt = -loss_j - loss_k

        # =========================
        # GAIN (scatter)
        # =========================
        collision_flat = collision_rate.flatten()

        values_low = collision_flat * self._frac_low_flat
        values_high = collision_flat * (1.0 - self._frac_low_flat)

        dndt_flat = dndt.reshape(-1)
        dndt_flat.scatter_add_(0, self._flat_idx_low, values_low)
        dndt_flat.scatter_add_(0, self._flat_idx_high, values_high)

        return dndt




    """
    def _coagulation_basic_loops(self, ns, K, coag_map):
        
        #Version avec boucles FOR - RÉFÉRENCE pour validation
        #Garde cette méthode pour debug!
        
        ncells, nclasses = ns.shape
        dndt = torch.zeros_like(ns)

        v = self.m_class / self.RHO_SOOT

        for icell in range(ncells):
            n = ns[icell, :]
            K_cell = K[icell, :, :]

            # PERTE
            for i in range(nclasses):
                loss_i = 0.0
                for j in range(nclasses):
                    loss_i += K_cell[i, j] * n[j]
                dndt[icell, i] -= n[i] * loss_i

            # GAIN
            for j in range(nclasses):
                for k in range(j, nclasses):
                    v_new = v[j] + v[k]

                    k_target_low = 0
                    for i in range(nclasses - 1):
                        if v[i] <= v_new < v[i + 1]:
                            k_target_low = i
                            break
                    else:
                        k_target_low = nclasses - 1

                    k_target_high = min(k_target_low + 1, nclasses - 1)

                    if k_target_low == k_target_high:
                        frac_low = 1.0
                    else:
                        v_low = v[k_target_low]
                        v_high = v[k_target_high]
                        frac_low = (v_high - v_new) / (v_high - v_low)
                        frac_low = max(0.0, min(1.0, frac_low))

                    if j == k:
                        rate = 0.5 * K_cell[j, k] * n[j] * n[k]
                    else:
                        rate = K_cell[j, k] * n[j] * n[k]

                    dndt[icell, k_target_low] += frac_low * rate
                    if k_target_high != k_target_low:
                        dndt[icell, k_target_high] += (1.0 - frac_low) * rate

        return dndt
        """