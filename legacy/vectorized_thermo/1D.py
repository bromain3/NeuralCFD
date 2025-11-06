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
import time
import thermo_lib as tl
# TODO: recode with shape of U being the same as the shape of Ys so that we can fully vectorize without transposing
#TODO: improve computations by computing all fields once befor each splitting operation so it is more computationally efficient even if the computations are quite quick after the first one
# or by not computing all fields whene one is enough

# ---------------------- Physical / numerical parameters ----------------------
# Domain
L = 1.0
ncells = 200
dx = L / ncells
x = (np.arange(ncells) + 0.5) * dx

# Thermodynamics params
y = tl.yamlParser(r'C:\Users\bromain\anaconda3\Lib\site-packages\cantera\data\gri30.yaml')
vec_th = y.populateVectorizedThermoObject()
nspecies = vec_th.nspecies         # number of mass-fraction species
species_mw = vec_th.molar_masses  # shape (nspecies,), kg/mol



# Numerical parameters
CFL = 0.9
final_time = 0.005  # seconds (short demo)
max_steps = 1000



# ------ Vectorized thermo wrapper utilities (pour remplacer Cantera) ------
def e_from_T(T, Y):
    # internal energy per mass e(T,Y) = h(T,Y) - R_spec * T
    h = vec_th.h_mix(T, Y)       # (N,)
    Rspec = vec_th.R_spec_from_Y(Y)     # (N,)
    return h - Rspec * T         # (N,)

def expand_T_guess(T_prev):
    """
    Expand cell-centered T guess (N,) to interface T guess (N+1,)
    by arithmetic averaging.
    """
    N = T_prev.shape[0]
    T_int = np.zeros(N+1)
    # interfaces internes : moyenne des cellules voisines
    T_int[1:N] = 0.5 * (T_prev[:-1] + T_prev[1:])
    # bords : extrapolation simple
    T_int[0] = T_prev[0]
    T_int[-1] = T_prev[-1]
    return T_int

def expand_T_guess_linear(T_prev):
    N = T_prev.shape[0]
    T_int = np.zeros(N+1)
    # extrapolation à gauche et droite
    T_int[0] = 2*T_prev[0] - T_prev[1]
    T_int[-1] = 2*T_prev[-1] - T_prev[-2]
    # moyennes internes
    T_int[1:N] = 0.5 * (T_prev[:-1] + T_prev[1:])
    return T_int

def solve_T_from_e_rho(e, rho, Y, T_guess=None, maxiter=40, tol=1e-4):
    """
    Solve for T given e (internal energy per mass), rho, Y.
    Vectorized Newton method with warm-start capability.
    """
    N = e.shape[0]

    if T_guess is None:
        # fallback guess: proportional to e/cv at 300K
        T0 = np.full(N, 300.0)
        cv0 = vec_th.cv_mix(T0, Y)
        T = np.maximum(300.0, e / (cv0 + 1e-12))
    else:
        T = T_guess.copy()
    Rspec = vec_th.R_spec_from_Y(Y)

    for i in range(maxiter):
        h = vec_th.h_mix(T, Y)
        f = h - Rspec * T - e
        err = np.max(np.abs(f))
        if err < tol:
            break
        cp = vec_th.cp_mix(T, Y)
        df = cp - Rspec
        df_safe = np.where(np.abs(df) < 1e-12, np.sign(df) * 1e-12 + 1e-12, df)
        T = T - f / df_safe
        T = np.clip(T, 1.0, 2e4)  # garde physique
    if i == maxiter - 1:
        bad_idx = np.where(np.abs(f) > tol)[0]
        if bad_idx.size > 0:
            bi = bad_idx[:10]
            print("  example bad idx:", bi)
            print("  e[bi]:", e[bi])
            print("  Rspec[bi]:", Rspec[bi])
            print("  T_guess[bi]:", T[bi])
            print("  cp[bi]:", cp[bi])
            print("  df[bi] (cp-Rspec):", df[bi])
        raise RuntimeError('Maximum number of iterations reached for temperature computation')
    return T



# ---------------------- Utility converters -----------------------------------
def conserved_to_primitive(U, T_prev=None):
    """
    U shape: (3+nspecies, ncells)
    Returns:
      rho (ncells,), u (ncells,), p (ncells,), T (ncells,),
      Ys (ncells, nspecies), cv (ncells,), gamma (ncells,)
    """
    nvars, N = U.shape
    rho = U[0, :].copy()
    inv_rho = 1.0 / rho
    u = (U[1, :] * inv_rho).copy()
    E = (U[2, :] * inv_rho).copy()  # total energy per mass
    kinetic = 0.5 * u ** 2
    e = E - kinetic  # internal energy per mass

    Ys = (U[3:3 + nspecies] * inv_rho).T  # shape (N, nspecies)

    # invert e->T vectorisé
    # use a reasonable initial guess per cell: max(300K, e/cv(300K))
    T = solve_T_from_e_rho(e, rho, Ys, T_guess=T_prev)

    Rspec = vec_th.R_spec_from_Y(Ys)
    p = rho * Rspec * T

    cv = vec_th.cv_mix(T, Ys)
    cp = vec_th.cp_mix(T, Ys)
    gamma = cp / cv
    a = np.sqrt(np.maximum(0.0, gamma * Rspec * T))

    return rho, u, p, T, Ys, cv, gamma, e, a

def primitive_to_conserved(rho, u, p, T, Ys):
    """
    rho (ncells,), u (ncells,), p (ncells,), T (ncells,), Ys (ncells, nspecies)
    returns U shape (3+nspecies, ncells)
    """
    N = rho.size
    # compute internal energy per mass directly: e = h - R_spec * T
    Rspec = vec_th.R_spec_from_Y(Ys)
    h = vec_th.h_mix(T, Ys)
    e = h - Rspec * T

    kinetic = 0.5 * u**2
    E = e + kinetic  # total energy per mass

    U = np.empty((3 + nspecies, N))
    U[0, :] = rho
    U[1, :] = rho * u
    U[2, :] = rho * E
    U[3:3+nspecies, :] = (Ys.T * rho)
    return U



# ---------------------- Rusanov flux for Euler + scalars ---------------------
def flux(U, Tprev=None):
    # compute flux vector for conserved variables for each cell center (so returns F at cells)
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)
    # conservative fluxes:
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = (rho * (e + 0.5 * u**2) + p) * u  # rho E * u + p u
    F[3:3+nspecies, :] = Ys.T * rho * u
    return F

def max_wave_speed(U, Tprev=None):
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)
    return np.max(np.abs(u) + a)

def max_wave_speed_vectorized(U, Tprev=None):
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)
    return np.abs(u) + a

def rusanov_flux(UL, UR, Tprev=None):
    # UL, UR are states at left and right of interface (shape: nvars)
    # We will vectorize this in the transport step.
    FL = flux(UL, Tprev)
    FR = flux(UR, Tprev)
    sL = max_wave_speed(UL, Tprev)
    sR = max_wave_speed(UR, Tprev)
    smax = np.maximum(sL, sR)
    # Rusanov: F = 0.5*(FL+FR) - 0.5*smax*(UR-UL)
    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL), smax

def rusanov_flux_vectorized(UL, UR, Tprev=None):
    # UL, UR are states at left and right of interface (shape: nvars)
    # We will vectorize this in the transport step.
    FL = flux(UL, Tprev)
    FR = flux(UR, Tprev)
    sL = max_wave_speed_vectorized(UL, Tprev)
    sR = max_wave_speed_vectorized(UR, Tprev)
    smax = np.maximum(sL, sR)
    # Rusanov: F = 0.5*(FL+FR) - 0.5*smax*(UR-UL)
    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL), np.max(smax)

# ---------------------- Transport step (finite volume) ----------------------
def transport_step(U, dt, Tprev=None, bc='periodic', inflow_state=None, vectorized=True):
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

    Tprev = expand_T_guess(Tprev)  # lowest order interpolation TODO: mettre un switch pour choisir l'ordre de la reconstruction
    # compute flux at interfaces
    if vectorized:
        F_int, smax_global = rusanov_flux_vectorized(UL_int, UR_int, Tprev)
    else:
        F_int = np.zeros_like(UL_int)
        smax_global = 0.0
        for i in range(N+1):
            Fi, smax = rusanov_flux(UL_int[:, i:i+1], UR_int[:, i:i+1], Tprev)
            F_int[:, i] = Fi[:, 0]
            smax_global = max(smax_global, smax)

    # update conserved: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    U_new = U.copy()
    U_new[:, :] = U[:, :] - (dt / dx) * (F_int[:, 1:] - F_int[:, :-1])
    return U_new, smax_global

# ---------------------- Chemistry step (calls user'db_gen NN) ----------------------
def chemistry_step_with_nn(U, dt, nn_predict, Tprev=None):
    # chemistry acts locally cell-by-cell. We keep rho and rho*u constant, update E and Ys
    rho = U[0].copy()
    u = U[1] / rho
    _, _, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)  # Ys shape (ncells, nspecies)
    # Call the NN: it should accept arrays and return arrays
    T_new, p_new, Ys_new = nn_predict(T, p, Ys, dt)
    # Sanity shapes
    assert T_new.shape == T.shape
    assert p_new.shape == p.shape
    assert Ys_new.shape == Ys.shape
    # Reconstruct conserved: keep rho and rho*u, update rho*E and rho*Y. it means no mass transfer between cells during chemestry, so constant volume
    # for consistency, we can recompute P:
    #Rspec_new = R_spec_from_Y(Ys_new)
    #p_new = rho * Rspec_new * T_new

    U_new = primitive_to_conserved(rho, u, p_new, T_new, Ys_new)
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
    rho0 = 0.1 + 0.0 * x
    u0 = 0.0 + 0.0 * x
    T0 = 300.0 + 0.0 * x #300.0 + 300.0 * np.exp(-((x - 0.5)**2) / (2 * 0.02**2))  # hot spot in center
    Ys0 = np.zeros((ncells, nspecies))
    #Ys0[:, 0] = 0.5 * np.exp(-((x - 0.5)**2) / (2 * 0.05**2)) + 0.25
    #Ys0[:, 1] = 1.0 - Ys0[:, 0]
    Ys0[:, 0] = 0.5
    Ys0[:, 1] = 1.0 - Ys0[:, 0]
    Rspec0 = vec_th.R_spec_from_Y(Ys0)
    p0 = rho0 * Rspec0 * T0

    U0 = primitive_to_conserved(rho0, u0, p0, T0, Ys0)
    return U0

# ---------------------- Time integration with Strang splitting ----------------
def run_solver(nn_predict, final_time, U0):
    U = U0.copy()
    t = 0.0
    step = 0
    history = {'t': [], 'T': [], 'Y0': [], 'rhoE':[]}

    _, _, _, T_prev, _, _, _, _, _ = conserved_to_primitive(U)

    while t < final_time and step < max_steps:
        _, _, _, T_prev, _, _, _, _, _ = conserved_to_primitive(U, T_prev)
        # compute stable dt by CFL
        smax = np.max(max_wave_speed_vectorized(U))
        dt_cfl = CFL * dx / (smax + 1e-12)
        dt = min(dt_cfl, final_time - t)

        # Strang splitting: half transport, full chemistry, half transport
        #U_half, s1 = transport_step(U, 0.5 * dt, T_prev, bc='flame1d', inflow_state=inflow_state)
        #U_chemed = chemistry_step_with_nn(U_half, dt, nn_predict, T_prev)
        #U_new, s2 = transport_step(U_chemed, 0.5 * dt, T_prev, bc='flame1d', inflow_state=inflow_state)

        U_new, s2 = transport_step(U, dt, T_prev, bc='flame1d', inflow_state=inflow_state)

        U = U_new
        t += dt
        step += 1

        if step % 10 == 0 or t >= final_time:
            rho, u, p, T_prev, Ys, _, _, _, _ = conserved_to_primitive(U, T_prev)
            history['t'].append(t)
            history['T'].append(T_prev.copy())
            history['Y0'].append(Ys[:, 0].copy())
            history['rhoE'].append(np.sum(U[2].copy()))
    return U, T_prev, history, step

# ---------------------- Run demo --------------------------------------------
rho0 = np.array([0.1])  # conditions d’entrée
u0 = np.array([1.0])
T0 = np.array([1700.0])
Ys0 = np.zeros((1, nspecies))
Ys0[:, 0] = 0.5
Ys0[:, 1] = 1.0 - Ys0[:, 0]


# compute p0 vectorisé
Rspec_inflow = vec_th.R_spec_from_Y(Ys0)
p0 = rho0 * Rspec_inflow * T0

inflow_state = primitive_to_conserved(rho0, u0, p0, T0, Ys0).flatten()

U0 = initial_condition()
t0 = time.time()
U_final, T_prev, history, steps = run_solver(dummy_nn_predict, final_time, U0)
print('simulation took %.10f seconds' % (time.time() - t0))
rho, u, p, T_final, Ys_final, _, _, _, _ = conserved_to_primitive(U_final, T_prev)

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

plt.figure(figsize=(8,4))
plt.plot(history['t'], history['rhoE'])
plt.xlabel("t")
plt.ylabel("rhoE")
plt.title("energy evolution")
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
