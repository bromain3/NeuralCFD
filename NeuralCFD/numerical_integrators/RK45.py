"""
Runge-Kutta-Fehlberg 4(5) integrator with batched GPU support in PyTorch.
Features:
- Explicit RK45 method with adaptive step size control
- Error estimation using embedded 4th and 5th order formulas
- Batched processing for (B, N) shaped systems on GPU/CPU
- Dense output interpolation (optional)
- Compatible with the backward Euler integrator structure

The integrator expects RHS f(x, t) to accept batched x of shape (B, N) and return (B, N).

Author: Based on scipy.integrate.solve_ivp RK45 implementation, adapted for PyTorch GPU batching
"""

from typing import Callable, Optional, Tuple
import torch


class RK45Integrator:
    """
    Dormand-Prince RK45 explicit integrator with adaptive step size.

    Butcher tableau for DOPRI5 (RK45):
    - 6 stages (7 function evaluations with FSAL)
    - Embedded 4th and 5th order methods for error estimation
    - Efficient for non-stiff problems
    """

    # Butcher tableau coefficients for Dormand-Prince RK45
    A = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    ]

    B = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]  # 5th order
    B_hat = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]  # 4th order

    C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1]

    def __init__(self,
                 f: Callable[[torch.Tensor, float], torch.Tensor],
                 device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None,
                 rtol: float = 1e-6,
                 atol: float = 1e-9,
                 max_step: float = float('inf'),
                 min_step: float = 0.0,
                 first_step: Optional[float] = None,
                 safety: float = 0.9,
                 min_factor: float = 0.2,
                 max_factor: float = 10.0,
                 verbose: bool = False):
        """
        Initialize RK45 integrator.

        Args:
            f: RHS function f(x, t) returning dx/dt, accepts (B, N) tensors
            device: torch device (cuda/cpu)
            dtype: torch dtype
            rtol: Relative tolerance for error control
            atol: Absolute tolerance for error control
            max_step: Maximum allowed step size
            min_step: Minimum allowed step size
            first_step: Initial step size (auto if None)
            safety: Safety factor for step size adjustment (< 1)
            min_factor: Minimum step size change factor
            max_factor: Maximum step size change factor
            verbose: Print integration progress
        """
        self.f = f
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype or torch.float32

        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.verbose = verbose

        # Convert Butcher tableau to tensors
        self._setup_coefficients()

        # Statistics
        self.n_steps = 0
        self.n_fevals = 0
        self.n_rejected = 0

    def _setup_coefficients(self):
        """Convert Butcher tableau to torch tensors."""
        self.C_tensor = torch.tensor(self.C, device=self.device, dtype=self.dtype)
        self.B_tensor = torch.tensor(self.B, device=self.device, dtype=self.dtype)
        self.B_hat_tensor = torch.tensor(self.B_hat, device=self.device, dtype=self.dtype)

        # A matrix is ragged, keep as list
        self.A_tensor = []
        for row in self.A:
            if row:
                self.A_tensor.append(torch.tensor(row, device=self.device, dtype=self.dtype))
            else:
                self.A_tensor.append(None)

    def _ensure_batch_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure x has batch dimension (B, N)."""
        if x.ndim == 1:
            return x.unsqueeze(0)
        return x

    def _estimate_initial_step(self, x0: torch.Tensor, t0: float) -> float:
        """
        Estimate initial step size using the algorithm from:
        E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary Differential Equations I"
        """
        with torch.no_grad():
            f0 = self.f(x0, t0)
            self.n_fevals += 1

            # Scale for error estimation
            scale = self.atol + self.rtol * torch.abs(x0)

            # d0 = norm(x0 / scale)
            d0 = torch.linalg.norm((x0 / scale).reshape(x0.shape[0], -1), dim=1).max().item()

            # d1 = norm(f0 / scale)
            d1 = torch.linalg.norm((f0 / scale).reshape(f0.shape[0], -1), dim=1).max().item()

            # Initial step estimate
            if d0 < 1e-5 or d1 < 1e-5:
                h0 = 1e-6
            else:
                h0 = 0.01 * d0 / d1

            # Perform one Euler step
            x1 = x0 + h0 * f0
            f1 = self.f(x1, t0 + h0)
            self.n_fevals += 1

            # d2 = norm((f1 - f0) / scale) / h0
            d2 = torch.linalg.norm(((f1 - f0) / scale).reshape(x0.shape[0], -1), dim=1).max().item() / h0

            # Refined estimate
            if max(d1, d2) <= 1e-15:
                h1 = max(1e-6, h0 * 1e-3)
            else:
                order = 5  # 5th order method
                h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))

            return min(100 * h0, h1, self.max_step)

    def _compute_stages(self, x: torch.Tensor, t: float, h: float) -> list:
        """Compute the 6 RK stages."""
        K = []

        # Stage 1 (reuse from previous step if FSAL)
        K.append(self.f(x, t))
        self.n_fevals += 1

        # Stages 2-6
        for i in range(1, 6):
            x_stage = x.clone()
            for j, a_ij in enumerate(self.A_tensor[i]):
                x_stage = x_stage + h * a_ij * K[j]

            t_stage = t + self.C_tensor[i] * h
            K.append(self.f(x_stage, t_stage))
            self.n_fevals += 1

        return K

    def _estimate_error(self, h: float, K: list) -> torch.Tensor:
        """
        Estimate local truncation error using embedded method.
        Returns error per batch element.
        """
        # 5th order solution
        x_5 = sum(h * b * k for b, k in zip(self.B_tensor, K))

        # 4th order solution (needs 7th stage)
        K_7 = K[0]  # For FSAL, K[0] of next step
        K_extended = K + [K_7]
        x_4 = sum(h * b * k for b, k in zip(self.B_hat_tensor, K_extended))

        # Error estimate
        error = x_5 - x_4

        return error

    def _compute_error_norm(self, x: torch.Tensor, x_new: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled error norm for each batch element.
        Returns (B,) tensor of error norms.
        """
        # Scale: atol + rtol * max(|x|, |x_new|)
        scale = self.atol + self.rtol * torch.max(torch.abs(x), torch.abs(x_new))

        # Norm per batch element
        error_scaled = error / scale
        error_norm = torch.linalg.norm(error_scaled.reshape(error_scaled.shape[0], -1), dim=1)

        # RMS norm (divide by sqrt(n))
        n = x.shape[1]
        error_norm = error_norm / (n ** 0.5)

        return error_norm

    def _update_step_size(self, h: float, error_norm: torch.Tensor) -> Tuple[float, bool]:
        """
        Update step size based on error estimate.

        Returns:
            new_step: Updated step size
            accept: Whether to accept the current step
        """
        # Use maximum error across batch
        error_max = error_norm.max().item()

        if error_max == 0:
            factor = self.max_factor
        else:
            # Standard step size control
            factor = self.safety * error_max ** (-1.0 / 5.0)  # order = 5
            factor = max(self.min_factor, min(factor, self.max_factor))

        accept = error_max <= 1.0

        if accept:
            if error_max == 0:
                h_new = h * self.max_factor
            else:
                h_new = h * min(self.max_factor, max(1.0, factor))
        else:
            h_new = h * factor

        # Enforce bounds
        h_new = max(self.min_step, min(self.max_step, h_new))

        return h_new, accept

    def step(self, x: torch.Tensor, t: float, h: float) -> Tuple[torch.Tensor, float, float]:
        """
        Perform one adaptive RK45 step.

        Args:
            x: Current state (B, N)
            t: Current time
            h: Attempted step size

        Returns:
            x_new: New state (B, N)
            t_new: New time
            h_new: Suggested next step size
        """
        x = self._ensure_batch_dim(x).to(device=self.device, dtype=self.dtype)

        max_attempts = 10
        for attempt in range(max_attempts):
            # Compute stages
            K = self._compute_stages(x, t, h)

            # 5th order solution
            x_new = x.clone()
            for b, k in zip(self.B_tensor, K):
                x_new = x_new + h * b * k

            # Error estimation (simplified - not using 7th stage)
            # For true error estimate, need to compute K_7 at (x_new, t+h)
            K_7 = self.f(x_new, t + h)
            self.n_fevals += 1
            K_extended = K + [K_7]

            x_4 = x.clone()
            for b, k in zip(self.B_hat_tensor, K_extended):
                x_4 = x_4 + h * b * k

            error = x_new - x_4
            error_norm = self._compute_error_norm(x, x_new, error)

            # Step size control
            h_new, accept = self._update_step_size(h, error_norm)

            if accept:
                self.n_steps += 1
                if self.verbose:
                    print(f"[RK45] t={t:.3e} -> {t + h:.3e}, h={h:.3e}, err={error_norm.max():.3e}, accepted")
                return x_new, t + h, h_new
            else:
                self.n_rejected += 1
                h = h_new
                if self.verbose:
                    print(f"[RK45] t={t:.3e}, h={h:.3e}, err={error_norm.max():.3e}, rejected (attempt {attempt + 1})")

        raise RuntimeError(f"RK45: Failed to find acceptable step after {max_attempts} attempts")

    def integrate(self, x0: torch.Tensor, t_span: Tuple[float, float],
                  t_eval: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate from t_span[0] to t_span[1].

        Args:
            x0: Initial state (B, N) or (N,)
            t_span: (t_start, t_end)
            t_eval: Specific times to evaluate at (optional)

        Returns:
            t_values: Times of evaluation (n_points,)
            x_values: States at evaluation times (n_points, B, N)
        """
        x0 = self._ensure_batch_dim(x0).to(device=self.device, dtype=self.dtype)
        t_start, t_end = t_span
        direction = 1 if t_end > t_start else -1

        # Reset statistics
        self.n_steps = 0
        self.n_fevals = 0
        self.n_rejected = 0

        # Initial step size
        if self.first_step is None:
            h = self._estimate_initial_step(x0, t_start)
        else:
            h = self.first_step
        h = direction * min(abs(h), self.max_step)

        # Storage for solution
        if t_eval is None:
            t_values = [t_start]
            x_values = [x0.cpu()]
        else:
            t_eval_list = t_eval.tolist() if isinstance(t_eval, torch.Tensor) else list(t_eval)
            t_values = []
            x_values = []
            eval_idx = 0

        # Integration loop
        t = t_start
        x = x0

        while direction * (t_end - t) > 0:
            # Don't overshoot
            h = direction * min(abs(h), abs(t_end - t))

            # Take step
            x_new, t_new, h_new = self.step(x, t, h)

            # Store or interpolate output
            if t_eval is None:
                t_values.append(t_new)
                x_values.append(x_new.cpu())
            else:
                # Store all evaluation points in [t, t_new]
                while eval_idx < len(t_eval_list) and direction * (t_eval_list[eval_idx] - t) >= 0 and direction * (
                        t_eval_list[eval_idx] - t_new) <= 0:
                    # Linear interpolation (could be improved with dense output)
                    alpha = (t_eval_list[eval_idx] - t) / (t_new - t)
                    x_interp = x + alpha * (x_new - x)
                    t_values.append(t_eval_list[eval_idx])
                    x_values.append(x_interp.cpu())
                    eval_idx += 1

            x = x_new
            t = t_new
            h = h_new

        if self.verbose:
            print(f"\n[RK45] Integration complete:")
            print(f"  Steps: {self.n_steps} (rejected: {self.n_rejected})")
            print(f"  Function evaluations: {self.n_fevals}")

        t_tensor = torch.tensor(t_values, device=self.device, dtype=self.dtype)
        x_tensor = torch.stack(x_values, dim=0).to(self.device)

        return t_tensor, x_tensor


# ------------------------ Example usage & tests -------------------------
if __name__ == '__main__':
    import time
    import math

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test 1: Linear ODE system (analytical solution exists)
    print("\n=== Test 1: Linear system ===")
    B = 1000
    N = 10
    torch.manual_seed(42)

    # Stable matrix
    A = torch.randn((B, N, N), device=device) * 0.1
    for b in range(B):
        A[b] = A[b] - 0.5 * torch.eye(N, device=device)


    def f_linear(x, t):
        return torch.einsum('bmn,bn->bm', A, x)


    x0 = torch.randn((B, N), device=device)
    t_span = (0.0, 1.0)

    integrator = RK45Integrator(f_linear, device=device, verbose=True, rtol=1e-6, atol=1e-9)

    t_start = time.perf_counter()
    t_vals, x_vals = integrator.integrate(x0, t_span)
    t_elapsed = time.perf_counter() - t_start

    print(f"Time elapsed: {t_elapsed:.3f} s")
    print(f"Final state shape: {x_vals[-1].shape}")
    print(f"Final state sample: {x_vals[-1][0, :3]}")

    # Test 2: Stiff nonlinear ODE (van der Pol oscillator, batched)
    print("\n=== Test 2: Van der Pol oscillator ===")
    mu_vdp = 10.0  # Stiffness parameter


    def f_vdp(x, t):
        # x[:, 0] = position, x[:, 1] = velocity
        dxdt = torch.zeros_like(x)
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = mu_vdp * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]
        # Fill remaining dims with zero
        return dxdt


    B_vdp = 100
    N_vdp = 2
    x0_vdp = torch.zeros((B_vdp, N_vdp), device=device)
    x0_vdp[:, 0] = 2.0  # Initial displacement
    x0_vdp[:, 1] = 0.0  # Initial velocity

    integrator_vdp = RK45Integrator(f_vdp, device=device, verbose=True,
                                    rtol=1e-5, atol=1e-8, max_step=0.1)

    t_span_vdp = (0.0, 10.0)
    t_eval_vdp = torch.linspace(0, 10, 100, device=device)

    t_start = time.perf_counter()
    t_vals_vdp, x_vals_vdp = integrator_vdp.integrate(x0_vdp, t_span_vdp, t_eval=t_eval_vdp)
    t_elapsed = time.perf_counter() - t_start

    print(f"Time elapsed: {t_elapsed:.3f} s")
    print(f"Output times: {len(t_vals_vdp)}")
    print(f"Final position (first batch): {x_vals_vdp[-1, 0, 0]:.3f}")

    print("\n=== Tests complete ===")