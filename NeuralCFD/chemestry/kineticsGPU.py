"""
cuda_chemistry.py — GPU-resident chemistry integration for NeuralCFD.

Implements kineticsGPU, a drop-in replacement for canteraChem that keeps
all data on the same CUDA device as the incoming torch tensors.

Constraints honoured:
  1. Zero CPU↔GPU transfers during the `computeIntegratedTPY` call.
  2. The kernel is launched on the exact device that owns the torch tensors
     (cuda:0, cuda:1, …).  The Numba CUDA context is switched accordingly.
  3. All persistent buffers (d_TC, integrator) are allocated once in __init__
     and reused every call to avoid allocation overhead.
  4. Only `cst_rho` (constant-volume) is implemented; `cst_p` raises
     NotImplementedError.

Dependencies
------------
  kineticsGPU   — your CUDA reaction-rate / integrator library
  numba.cuda
  torch
  numpy  (only for one-time init, never inside the hot path)

Usage
-----
    from NeuralCFD.chemestry.cuda_chemistry import kineticsGPU

    chem = kineticsGPU(
        yaml_file   = "gri30.yaml",
        ncells      = grid.ncells,
        device      = grid.cells.T.device,   # e.g. torch.device("cuda:0")
        newton_tol     = 1e-6,
        newton_maxiter = 20,
        gmres_restart  = 20,
        gmres_tol      = 1e-6,
        gmres_maxiter  = 50,
        dtype          = torch.float64,
    )

    # inside your time-loop:
    T_new, p_new, Ys_new = chem.computeIntegratedTPY(grid, dt, "cst_rho")
"""

import numpy as np
import torch
import numba.cuda as numba_cuda

import kineticsGPU.torch.thermo as yaml_thermo
from kineticsGPU.torch.reactionParser import parseReactions, build_reactions
from kineticsGPU.cuda.mechanism.mechanism_data import build_mechanism_data_from_torch
from kineticsGPU.cuda.integrator.cuda_integrator_pb import (
    CUDABackwardEulerIntegratorPB,
)

from NeuralCFD.chemestry.DummyChemistry import DummyChemistry


# ---------------------------------------------------------------------------
# Helpers — torch ↔ Numba device-array bridge (zero-copy on same device)
# ---------------------------------------------------------------------------

def _torch_to_numba(t: torch.Tensor):
    """
    Return a Numba device array that *shares* the underlying CUDA memory with
    the torch tensor `t`.  No data is moved; this is a pointer cast.
    `t` must be contiguous and on a CUDA device.
    """
    assert t.is_cuda and t.is_contiguous(), (
        "Tensor must be contiguous and on a CUDA device."
    )
    # __cuda_array_interface__ is the standard zero-copy bridge
    return numba_cuda.as_cuda_array(t)


def _numba_to_torch(arr, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return a torch tensor sharing memory with a Numba device array `arr`.
    No data is moved.
    """
    return torch.as_tensor(arr, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class kineticsGPU(DummyChemistry):
    """
    GPU-resident chemistry integrator for NeuralCFD.

    Parameters
    ----------
    yaml_file : str
        Path to the Cantera YAML mechanism file (e.g. ``"gri30.yaml"``).
    ncells : int
        Number of computational cells (= number of independent reactors).
    device : torch.device | str
        Target CUDA device, e.g. ``torch.device("cuda:0")``.
        Must match the device of the tensors passed to
        ``computeIntegratedTPY``.
    newton_tol : float
        Absolute tolerance for the Newton solver (default 1e-6).
    newton_maxiter : int
        Maximum Newton iterations per time step (default 20).
    gmres_restart : int
        GMRES restart parameter (default 20).
    gmres_tol : float
        GMRES relative tolerance (default 1e-6).
    gmres_maxiter : int
        Maximum GMRES iterations (default 50).
    dtype : torch.dtype
        Floating-point precision; must be ``torch.float64`` or
        ``torch.float32`` (default ``torch.float64``).
    """

    def __init__(
        self,
        yaml_file: str,
        ncells: int,
        device: torch.device | str = "cuda:0",
        newton_tol: float = 1e-12,
        newton_maxiter: int = 200,
        gmres_restart: int = 20,
        gmres_tol: float = 1e-12,
        gmres_maxiter: int = 50,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.ncells = ncells
        self.device = torch.device(device)
        self.torch_dtype = dtype
        self.np_dtype = np.float64 if dtype == torch.float64 else np.float32

        if not self.device.type == "cuda":
            raise ValueError(
                f"kineticsGPU requires a CUDA device, got '{self.device}'."
            )

        # Derive the integer device index (0, 1, …) so we can select the
        # matching Numba CUDA context.
        self._cuda_index: int = self.device.index if self.device.index is not None else 0

        # ── Select the correct Numba CUDA device once ──────────────────
        # All subsequent Numba operations (to_device, kernel launches) are
        # bound to this context automatically because Numba uses the active
        # CUDA context, which we set here and keep for the lifetime of this
        # object.
        numba_cuda.select_device(self._cuda_index)

        # ── Load mechanism ─────────────────────────────────────────────
        torch_device_str = str(self.device)   # e.g. "cuda:0"

        y      = yaml_thermo.yamlParser(yaml_file)
        vec_th = y.populateVectorizedThermoObject(
            dtype=dtype, device=torch_device_str
        )
        rxns = parseReactions(yaml_file)
        r_e, r_tb, r_f, _system = build_reactions(
            vec_th.species, rxns, vec_th, torch_device_str, dtype
        )

        self.mech = build_mechanism_data_from_torch(
            r_e, r_tb, r_f, vec_th, dtype=self.np_dtype
        )
        self.n_species: int = self.mech.n_species
        self.species_names = self.mech.thermo.species_names  # list[str]

        # ── Pre-allocate persistent GPU state buffer (torch) ───────────
        # Layout: TC[:, 0] = T (K),  TC[:, 1:] = C (mol/m³)
        self._TC = torch.zeros(
            (ncells, self.n_species + 1),
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.converged = torch.zeros((ncells, ), dtype=self.torch_dtype, device=self.device)

        # ── Build integrator (triggers Numba JIT on first .step call) ──
        self._integrator = CUDABackwardEulerIntegratorPB(
            mechanism      = self.mech,
            n_reactors     = ncells,
            newton_tol     = newton_tol,
            newton_maxiter = newton_maxiter,
            gmres_restart  = gmres_restart,
            gmres_tol      = gmres_tol,
            gmres_maxiter  = gmres_maxiter,
            dtype          = self.np_dtype,
        )

        # ── Warm-up: force JIT compilation now, not at first CFD step ──
        #_wm_TC = numba_cuda.as_cuda_array(self._TC)
        #_wm_TC = self._integrator.step(_wm_TC, h=1e-10)
        #numba_cuda.synchronize()
        # Reset buffer to zero after warm-up
        self._TC.zero_()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def computeIntegratedTPY(
        self,
        grid,
        dt: float,
        reactor_model: str = "cst_rho",
    ):
        """
        Integrate chemistry for `dt` seconds using the CUDA backward-Euler
        integrator.

        Parameters
        ----------
        grid : CFD grid object
            Must expose ``grid.cells.T``, ``grid.cells.rho``,
            ``grid.cells.Ys`` as CUDA torch tensors of shape ``(ncells,)``
            / ``(ncells, n_species)``.
        dt : float
            Chemical time step in seconds.
        reactor_model : str
            Only ``"cst_rho"`` (constant-volume / constant-density) is
            implemented.  Passing anything else raises ``NotImplementedError``.

        Returns
        -------
        T_new : torch.Tensor   shape (ncells,)
        p_new : torch.Tensor   shape (ncells,)
        Ys_new : torch.Tensor  shape (ncells, n_species)

        Notes
        -----
        * No CPU↔GPU transfers occur inside this method.
        * All tensors are returned on the same device as ``grid.cells.T``.
        """
        if reactor_model != "cst_rho":
            raise NotImplementedError(
                f"kineticsGPU only supports 'cst_rho' reactor model for now; "
                f"got '{reactor_model}'."
            )

        # ── Validate device consistency ────────────────────────────────
        grid_device = grid.cells.T.device
        if grid_device != self.device:
            raise RuntimeError(
                f"kineticsGPU was initialised on {self.device} but the grid "
                f"tensors are on {grid_device}.  Devices must match."
            )
        T   = grid.cells.T    # (ncells,)       K
        rho = grid.cells.rho  # (ncells,)       kg/m³
        Ys  = grid.cells.Ys   # (ncells, n_sp)  mass fractions

        # ── Convert mass fractions → molar concentrations (mol/m³) ────
        # C_k = rho * Y_k / W_k  [mol/m³]
        # Molecular weights: retrieve from mechanism as a torch tensor on
        # the correct device (shape: (n_sp,)).
        W = self._get_molecular_weights()   # (n_sp,)  kg/mol

        # (ncells, n_sp) — all on GPU, no transfer
        C = rho.unsqueeze(1) * Ys / W.unsqueeze(0)   # mol/m³

        # ── Fill persistent TC buffer in-place (no new allocation) ────
        self._TC[:, 0]  = T
        self._TC[:, 1:] = C
        # ── Zero-copy view as Numba device array ───────────────────────
        n_cells = self._TC.shape[0]
        TC_initial = self._TC.clone()

        h_full = np.full(n_cells, dt, dtype=np.float64)
        d_TC = numba_cuda.as_cuda_array(self._TC)
        d_TC, d_converged = self._integrator.step(d_TC, h=h_full)

        result_torch = torch.as_tensor(d_TC, device=self.device)
        converged_torch = torch.as_tensor(d_converged, device=self.device)

        failed = (converged_torch != 1)
        if failed.any():
            # Repartir de l'état initial pour les réacteurs échoués
            if result_torch.data_ptr() != self._TC.data_ptr():
                self._TC.copy_(result_torch)
            self._TC[failed] = TC_initial[failed]

            failed_np = failed.cpu().numpy()
            h_fourth = np.where(failed_np, dt / 4, 0.0)

            d_TC = numba_cuda.as_cuda_array(self._TC)
            d_TC, d_converged = self._integrator.step(d_TC, h=h_fourth)

            d_TC, d_converged = self._integrator.step(d_TC, h=h_fourth)

            d_TC, d_converged = self._integrator.step(d_TC, h=h_fourth)

            d_TC, d_converged = self._integrator.step(d_TC, h=h_fourth)


            converged_torch = torch.as_tensor(d_converged, device=self.device)
            result_torch = torch.as_tensor(d_TC, device=self.device)

            still_failed = (converged_torch != 1) & failed
            if still_failed.any():
                raise ValueError(
                    f'Reactors {still_failed.nonzero().flatten()} not converged '
                    f'after chemical iteration'
                )

        if result_torch.data_ptr() != self._TC.data_ptr():
            self._TC.copy_(result_torch)

        T_new = self._TC[:, 0].clone()
        C_new = self._TC[:, 1:].clone()

        # ── Convert molar concentrations back to mass fractions ────────
        # Y_k = C_k * W_k / rho_new
        # At constant volume rho is constant, so rho_new = rho.
        mass_new   = C_new * W.unsqueeze(0)          # (ncells, n_sp) kg/m³
        rho_new    = mass_new.sum(dim=1, keepdim=True)   # (ncells, 1)  # TODO: changer ça et calculer rho avant juste pour être sur même si normalement ça ne change rien
        Ys_new     = mass_new / rho_new.clamp(min=1e-30)

        # ── Recompute pressure from ideal-gas law ──────────────────────
        # p = rho_new * R_spec * T_new
        # R_spec = sum_k (Y_k * R_u / W_k)  where R_u = 8314.46 J/kmol/K
        R_u    = 8.314_462_618_153_24   # J/mol/K
        R_spec = (Ys_new / W.unsqueeze(0)).sum(dim=1) * R_u   # (ncells,) J/kg/K
        p_new  = rho_new.squeeze(1) * R_spec * T_new           # (ncells,) Pa
        return T_new, p_new, Ys_new

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_molecular_weights(self) -> torch.Tensor:
        """
        Return molecular weights as a torch tensor on ``self.device``.

        The result is cached after the first call so no allocation occurs
        in the hot path.
        """
        if not hasattr(self, "_W"):
            # mech.thermo.molecular_weights is a numpy array in kg/mol
            W_np = np.asarray(
                self.mech.thermo.molar_masses, dtype=self.np_dtype
            )
            self._W = torch.tensor(
                W_np, dtype=self.torch_dtype, device=self.device
            )
        return self._W