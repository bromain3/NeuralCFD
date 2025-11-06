# Code for a minimal 1D finite-volume solver (Euler + passive scalars Y_i) with operator splitting (Strang)
# The "chemistry" step is *not* source-term based: it calls a user-provided function `nn_predict(T, P, Y, dt)`
# which returns the updated (T_new, P_new, Y_new) after time dt for each cell.
#
# Assumptions/simplifications:
# - Ideal gas with constant gamma and gas constant R_spec (same for all species)
# - Conserved variables: [rho, rho*u, rho*E, rho*Y1, rho*Y2, ...]
# - Chemistry step keeps density rho and momentum rho*u constant (no mass or momentum exchange during chemistry)
# - The NN must accept numpy arrays and return numpy arrays of the same shape.
#
# How to plug your network:
# - Replace the dummy_nn_predict(...) function by a wrapper that calls your NN (PyTorch/TF/ONNX).
# - The wrapper should accept vectors T, P, Y (shape: (ncells,), (ncells,), (ncells, nspecies))
#   and return T_new, P_new, Y_new of the same shapes.
#
# The script demonstrates a short run with a dummy NN (identity) and plots T and Y.
# You can easily replace the dummy NN with calls to your model.

import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Physical / numerical parameters ----------------------
gamma = 1.4          # ratio of specific heats
R_spec = 287.0       # specific gas constant J/(kg K) (for air-like mixture)
cv = R_spec / (gamma - 1)
nspecies = 2         # number of mass-fraction species

# Domain
L = 1.0
ncells = 200
dx = L / ncells
x = (np.arange(ncells) + 0.5) * dx

# Numerical parameters
CFL = 0.5
final_time = 0.005  # seconds (short demo)
max_steps = 10000

# ---------------------- Utility converters -----------------------------------
def conserved_to_primitive(U):
    # U shape: (5+ns-1, ncells) or a 2D array with shape (nvars, ncells)
    # We'll assume U is shape (nvars, ncells)
    rho = U[0].copy()
    u = U[1] / rho
    E = U[2] / rho
    kinetic = 0.5 * u**2
    e = E - kinetic  # internal energy per unit mass
    T = e / cv
    p = (gamma - 1.0) * rho * e
    Ys = (U[3:3+nspecies] / rho)  # shape (nspecies, ncells)
    return rho, u, p, T, Ys.T  # Ys returned shape (ncells, nspecies)

def primitive_to_conserved(rho, u, p, T, Ys):
    # Ys shape: (ncells, nspecies)
    E = cv * T + 0.5 * u**2
    U = np.zeros((3 + nspecies, rho.size))
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * E
    for i in range(nspecies):
        U[3+i] = rho * Ys[:, i]
    return U

# ---------------------- Rusanov flux for Euler + scalars ---------------------
def flux(U):
    # compute flux vector for conserved variables for each cell center (so returns F at cells)
    rho, u, p, T, Ys = conserved_to_primitive(U)
    # conservative fluxes:
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = (rho * (cv * T + 0.5 * u**2) + p) * u  # rho E * u + p u
    for i in range(nspecies):
        F[3 + i] = rho * Ys[:, i] * u
    return F

def max_wave_speed(U):
    rho, u, p, T, Ys = conserved_to_primitive(U)
    a = np.sqrt(gamma * p / rho)
    return np.max(np.abs(u) + a)

def rusanov_flux(UL, UR):
    # UL, UR are states at left and right of interface (shape: nvars)
    # We will vectorize this in the transport step.
    FL = flux(UL)
    FR = flux(UR)
    sL = max_wave_speed(UL)
    sR = max_wave_speed(UR)
    smax = np.maximum(sL, sR)
    # Rusanov: F = 0.5*(FL+FR) - 0.5*smax*(UR-UL)
    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL), smax

# ---------------------- Transport step (finite volume) ----------------------
def transport_step(U, dt, bc='periodic', inflow_state=None):
    """
    One explicit finite-volume step using Rusanov flux.
    bc : 'periodic' or 'flame1d'
    inflow_state : array of shape (nvars,) giving the conserved inflow state (only used if bc='flame1d')
    """
    # one explicit finite-volume step using Rusanov flux
    nvars, N = U.shape
    # apply periodic BC via ghost cells
    # build UL and UR at interfaces: there are N+1 interfaces, index i between cell i-1 and i
    # We'll create arrays UL_int, UR_int with shape (nvars, N+1)
    # There are N+1 interfaces indexed 0..N
    # interface i sits between cell i-1 (left) and cell i (right)
    UL_int = np.zeros((nvars, N + 1))
    UR_int = np.zeros((nvars, N + 1))
    # interior interfaces i=1..N-1: left is cell i-1, right is cell i
    # interior interfaces i = 1 .. N-1:
    # left state = U[:, i-1], right state = U[:, i]
    UL_int[:, 1:N] = U[:, 0:N - 1]
    UR_int[:, 1:N] = U[:, 1:N]
    if bc == 'periodic':
        # interface 0 : between cell N-1 (left) and cell 0 (right)
        UL_int[:, 0] = U[:, -1]
        UR_int[:, 0] = U[:, 0]
        # interface N : between cell N-1 (left) and cell 0 (right) as well (periodic wrap)
        # here we set the ghost on the right of the last cell equal to the first cell (periodic)
        UL_int[:, -1] = U[:, -1]
        UR_int[:, -1] = U[:, 0]
    elif bc == 'flame1d':
        if inflow_state is None:
            raise ValueError("Need inflow_state for bc='flame1d'")

        # left boundary (inflow) : ghost cell fixed to inflow state
        # left boundary (interface 0): left=ghost(inflow), right=cell 0
        UL_int[:, 0] = inflow_state
        UR_int[:, 0] = U[:, 0]

        # right boundary (outflow) : ghost cell = last interior cell (zero-gradient)
        # right boundary (interface N): left = last cell (N-1), right = ghost = copy of last cell => zero gradient
        UL_int[:, -1] = U[:, -1]
        UR_int[:, -1] = U[:, -1]

    else:
        raise NotImplementedError("Only 'periodic' or 'flame1d' BC implemented")

    # compute flux at interfaces
    F_int = np.zeros_like(UL_int)
    smax_global = 0.0
    for i in range(N+1):
        Fi, smax = rusanov_flux(UL_int[:, i:i+1], UR_int[:, i:i+1])
        F_int[:, i] = Fi[:, 0]
        smax_global = max(smax_global, smax)

    # update conserved: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    U_new = U.copy()
    U_new[:, :] = U[:, :] - (dt / dx) * (F_int[:, 1:] - F_int[:, :-1])
    return U_new, smax_global

# ---------------------- Chemistry step (calls user'db_gen NN) ----------------------
def chemistry_step_with_nn(U, dt, nn_predict):
    # chemistry acts locally cell-by-cell. We keep rho and rho*u constant, update E and Ys
    rho = U[0].copy()
    u = U[1] / rho
    _, _, p, T, Ys = conserved_to_primitive(U)  # Ys shape (ncells, nspecies)
    # Call the NN: it should accept arrays and return arrays
    T_new, p_new, Ys_new = nn_predict(T, p, Ys, dt)
    # Sanity shapes
    assert T_new.shape == T.shape
    assert p_new.shape == p.shape
    assert Ys_new.shape == Ys.shape
    # Reconstruct conserved: keep rho and rho*u, update rho*E and rho*Y
    E_new = cv * T_new + 0.5 * u**2  # per mass
    U_new = np.zeros_like(U)
    U_new[0] = rho
    U_new[1] = rho * u
    U_new[2] = rho * E_new
    for i in range(nspecies):
        U_new[3 + i] = rho * Ys_new[:, i]
    return U_new

# ---------------------- Dummy NN / user should replace ------------------------
def dummy_nn_predict(T, p, Ys, dt):
    # Identity model (no change) but adds a tiny relaxation for demonstration
    # T, p: shape (ncells,), Ys: shape (ncells, nspecies)
    # Return same shapes
    # Example: small exponential relaxation towards a reference T_ref in hot zone
    T_ref = 300.0
    tau = 1e-3  # relaxation time (db_gen)
    T_new = T + (T_ref - T) * (1 - np.exp(-dt / tau))
    # Keep pressure consistent with ideal gas p = rho R T? In chemistry step we let NN provide p.
    # Here we recompute p using ideal gas (user NN could return different p).
    # We'll just compute p_new from T_new (assumes same rho and R_spec).
    # But p input was available; if you want to let NN control p independently, modify accordingly.
    p_new = p * (T_new / (T + 1e-12))
    Ys_new = Ys.copy()
    # small species consumption in center
    Ys_new[:, 0] = Ys[:, 0] * np.exp(-dt * 50.0 * np.exp(-((x-0.5)**2)/(2*0.02**2)))
    Ys_new[:, 1] = 1.0 - Ys_new[:, 0]
    return T_new, p_new, Ys_new

# ---------------------- Initial condition -----------------------------------
def initial_condition():
    rho0 = 1.0 + 0.0 * x
    u0 = 0.0 + 0.0 * x
    T0 = 300.0 + 300.0 * np.exp(-((x - 0.5)**2) / (2 * 0.02**2))  # hot spot in center
    Ys0 = np.zeros((ncells, nspecies))
    Ys0[:, 0] = 0.5 * np.exp(-((x - 0.5)**2) / (2 * 0.05**2)) + 0.25
    Ys0[:, 1] = 1.0 - Ys0[:, 0]
    p0 = rho0 * R_spec * T0
    U0 = primitive_to_conserved(rho0, u0, p0, T0, Ys0)
    return U0

# ---------------------- Time integration with Strang splitting ----------------
def run_solver(nn_predict, final_time, U0):
    U = U0.copy()
    t = 0.0
    step = 0
    history = {'t': [], 'T': [], 'Y0': []}
    while t < final_time and step < max_steps:
        # compute stable dt by CFL
        smax = max_wave_speed(U)
        dt_cfl = CFL * dx / (smax + 1e-12)
        dt = min(dt_cfl, final_time - t)

        # Strang splitting: half transport, full chemistry, half transport
        U_half, s1 = transport_step(U, 0.5 * dt, bc='flame1d', inflow_state=inflow_state)
        U_chemed = chemistry_step_with_nn(U_half, dt, nn_predict)
        U_new, s2 = transport_step(U_chemed, 0.5 * dt, bc='flame1d', inflow_state=inflow_state)

        U = U_new
        t += dt
        step += 1

        if step % 10 == 0 or t >= final_time:
            rho, u, p, T, Ys = conserved_to_primitive(U)
            history['t'].append(t)
            history['T'].append(T.copy())
            history['Y0'].append(Ys[:, 0].copy())
    return U, history, step

# ---------------------- Run demo --------------------------------------------
rho0 = np.array([1.0])  # conditions d’entrée
u0 = np.array([1.0])
T0 = np.array([1000.0])
p0 = rho0 * R_spec * T0
Ys0 = np.zeros((1, nspecies))
Ys0[0, 0] = 0.5
Ys0[0, 1] = 1.0 - Ys0[0, 0]
inflow_state = primitive_to_conserved(rho0, u0, p0, T0, Ys0).flatten()

U0 = initial_condition()
U_final, history, steps = run_solver(dummy_nn_predict, final_time, U0)
rho, u, p, T_final, Ys_final = conserved_to_primitive(U_final)

print(f"Finished in {steps} steps, final time approx {history['t'][-1]:.6e} s")

# Plot final T and Y0 profiles
plt.figure(figsize=(8,4))
plt.plot(x, T_final)
plt.xlabel("x")
plt.ylabel("Temperature (K)")
plt.title("Final Temperature profile")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(x, Ys_final[:,0])
plt.xlabel("x")
plt.ylabel("Y[0]")
plt.title("Final species mass fraction Y0")
plt.tight_layout()
plt.show()

# Also show evolution at a few saved times
times = history['t']
if len(times) > 1:
    plt.figure(figsize=(8,4))
    for i, T_snap in enumerate(history['T']):
        plt.plot(x, T_snap, label=f"t={times[i]:.2e}")
    plt.xlabel("x"); plt.ylabel("T (K)"); plt.title("Temperature snapshots")
    #plt.legend()
    plt.tight_layout()
    plt.show()
