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
import cantera as ct

# TODO: recode with shape of U being the same as the shape of Ys so that we can fully vectorize without transposing

# ---------------------- Physical / numerical parameters ----------------------
# Domain
L = 1.0
ncells = 200
dx = L / ncells
x = (np.arange(ncells) + 0.5) * dx

# Thermodynamics params
gas = ct.Solution('gri30.yaml')
nspecies = len(gas.species_names)         # number of mass-fraction species



# Numerical parameters
CFL = 0.9
final_time = 0.005  # seconds (short demo)
max_steps = 10000

# ---------------------- Utility converters -----------------------------------
def conserved_to_primitive(U):
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
    E = (U[2, :] * inv_rho).copy()   # total energy per mass
    kinetic = 0.5 * u**2
    e = E - kinetic  # internal energy per mass

    Ys = np.zeros((N, nspecies))
    for i in range(N):
        Ys[i, :] = (U[3:3+nspecies, i] * inv_rho[i])

    T = np.zeros(N)
    p = np.zeros(N)
    cv = np.zeros(N)
    cp = np.zeros(N)
    gamma = np.zeros(N)
    a = np.zeros(N)

    # Set Cantera state per-cell using internal energy & specific volume & mass fractions:
    for i in range(N):
        # gas.UVY expects (u (int energy), v (specific volume), Y)
        # u = e[i], v = 1/rho[i]
        gas.UVY = e[i], inv_rho[i], Ys[i, :]
        T[i] = gas.T
        p[i] = gas.P
        cv[i] = gas.cv
        cp[i] = gas.cp
        gamma[i] = cp[i] / cv[i]
        a[i] = gas.sound_speed

    return rho, u, p, T, Ys, cv, gamma, e, a

def primitive_to_conserved(rho, u, p, T, Ys):
    """
    rho (ncells,), u (ncells,), p (ncells,), T (ncells,), Ys (ncells, nspecies)
    returns U shape (3+nspecies, ncells)
    """
    N = rho.size
    e = np.zeros(N)
    # Use Cantera to get internal energy per mass from (T,p,Y)
    for i in range(N):
        gas.TPY = T[i], p[i], Ys[i, :]
        e[i] = gas.int_energy_mass

    kinetic = 0.5 * u**2
    E = e + kinetic  # total energy per mass

    U = np.empty((3 + nspecies, N))
    U[0, :] = rho
    U[1, :] = rho * u
    U[2, :] = rho * E
    U[3:3+nspecies, :] = (Ys.T * rho)
    return U



# ---------------------- Rusanov flux for Euler + scalars ---------------------
def flux(U):
    # compute flux vector for conserved variables for each cell center (so returns F at cells)
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U)
    # conservative fluxes:
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = (rho * (e + 0.5 * u**2) + p) * u  # rho E * u + p u
    F[3:3+nspecies, :] = Ys.T * rho * u
    return F

def max_wave_speed(U):
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U)
    return np.max(np.abs(u) + a)

def max_wave_speed_vectorized(U):
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U)
    return np.abs(u) + a

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

def rusanov_flux_vectorized(UL, UR):
    # UL, UR are states at left and right of interface (shape: nvars)
    # We will vectorize this in the transport step.
    FL = flux(UL)
    FR = flux(UR)
    sL = max_wave_speed_vectorized(UL)
    sR = max_wave_speed_vectorized(UR)
    smax = np.maximum(sL, sR)
    # Rusanov: F = 0.5*(FL+FR) - 0.5*smax*(UR-UL)
    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL), np.max(smax)

# ---------------------- Transport step (finite volume) ----------------------
def transport_step(U, dt, bc='periodic', inflow_state=None, vectorized=True):
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
    if vectorized:
        F_int, smax_global = rusanov_flux_vectorized(UL_int, UR_int)
    else:
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
def chemistry_step_with_nn(U, dt, gas):
    rho = U[0].copy()
    u = U[1] / rho
    _, _, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U)

    # Appel à Cantera
    T_new, p_new, Ys_new = cantera_predict(T, p, Ys, dt, gas)

    # Reconstruction
    U_new = primitive_to_conserved(rho, u, p_new, T_new, Ys_new)
    return U_new

# ---------------------- Dummy NN / user should replace ------------------------
def cantera_predict(T, p, Ys, dt, gas):
    """
    Effectue l’intégration chimique locale via Cantera.

    Parameters
    ----------
    T : ndarray (ncells,)
        Température initiale (K)
    p : ndarray (ncells,)
        Pression initiale (Pa)
    Ys : ndarray (ncells, nspecies)
        Fractions massiques initiales
    dt : float
        Pas de temps (db_gen)
    gas : ct.Solution
        Objet Cantera avec le mécanisme chimique

    Returns
    -------
    T_new : ndarray (ncells,)
    p_new : ndarray (ncells,)
    Ys_new : ndarray (ncells, nspecies)
    """
    ncells, nspecies = Ys.shape
    T_new = np.zeros_like(T)
    p_new = np.zeros_like(p)
    Ys_new = np.zeros_like(Ys)

    for i in range(ncells):
        # Initialiser l’état du gaz dans la cellule i
        gas.TPY = T[i], p[i], Ys[i, :]

        # Créer un réacteur constant-volume
        reactor = ct.IdealGasReactor(gas, energy='on')
        sim = ct.ReactorNet([reactor])

        # Avancer en temps
        sim.advance(dt)

        # Récupérer les valeurs mises à jour
        T_new[i] = reactor.T
        p_new[i] = reactor.thermo.P
        Ys_new[i, :] = reactor.thermo.Y

    return T_new, p_new, Ys_new

# ---------------------- Initial condition -----------------------------------
def initial_condition():
    rho0 = 1.0 + 0.0 * x
    u0 = 0.1 + 0.0 * x
    T0 = 500.0 + 0.0 * x
    Ys0 = np.zeros((ncells, nspecies))
    Ys0[:, gas.species_names.index('CH4')] = 0.05483547715890419 + 0.0 * x
    Ys0[:, gas.species_names.index('N2')] = 0.7263764388501015 + 0.0 * x
    Ys0[:, gas.species_names.index('O2')] = 0.21878808399099442 + 0.0 * x

    p0 = np.zeros_like(rho0)  #TODO: define it only once for perf
    for i in range(len(rho0)):
        gas.TDY = T0[i], rho0[i], Ys0[i]
        p0[i] = gas.P

    U0 = primitive_to_conserved(rho0, u0, p0, T0, Ys0)
    return U0

# ---------------------- Time integration with Strang splitting ----------------
def run_solver(gas, final_time, U0):
    U = U0.copy()
    t = 0.0
    step = 0
    history = {'t': [], 'T': [], 'T_x0': [], 'Y0': [], 'rhoE':[]}
    while t < final_time and step < max_steps:
        # compute stable dt by CFL
        #smax = np.max(max_wave_speed_vectorized(U))
        #dt_cfl = CFL * dx / (smax + 1e-12)
        #dt = min(dt_cfl, final_time - t)
        dt = 1e-6

        # Strang splitting: half transport, full chemistry, half transport
        U_half, s1 = transport_step(U, 0.5 * dt, bc='flame1d', inflow_state=inflow_state)
        U_chemed = chemistry_step_with_nn(U_half, dt, gas)
        U_new, s2 = transport_step(U_chemed, 0.5 * dt, bc='flame1d', inflow_state=inflow_state)

        U = U_new
        t += dt
        step += 1

        if step % 10 == 0 or t >= final_time:
            print(step)
            rho, u, p, T, Ys, _, _, _, _ = conserved_to_primitive(U)
            history['t'].append(t)
            history['T'].append(T.copy())
            history['T_x0'].append(T.copy()[0])
            history['Y0'].append(Ys[:, 0].copy())
            history['rhoE'].append(np.sum(U[2].copy()))
    return U, history, step

# ---------------------- Run demo --------------------------------------------
rho0 = np.array([1.0])  # conditions d’entrée
u0 = np.array([0.1])
T0 = np.array([1700.0])
Ys0 = np.zeros((1, nspecies))
Ys0[:, gas.species_names.index('CH4')] = 0.05483547715890419
Ys0[:, gas.species_names.index('N2')] = 0.7263764388501015
Ys0[:, gas.species_names.index('O2')] = 0.21878808399099442

p0 = np.zeros_like(rho0)  # TODO: define it only once for perf
for i in range(len(rho0)):
    gas.TDY = T0[i], rho0[i], Ys0[i]
    p0[i] = gas.P

inflow_state = primitive_to_conserved(rho0, u0, p0, T0, Ys0).flatten()

U0 = initial_condition()
t0 = time.time()
U_final, history, steps = run_solver(gas, final_time, U0)
print('simulation took %.10f seconds' % (time.time() - t0))
rho, u, p, T_final, Ys_final, _, _, _, _ = conserved_to_primitive(U_final)

print(f"Finished in {steps} steps, final time approx {history['t'][-1]:.6e} db_gen")

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

plt.figure(figsize=(8,4))
plt.plot(history['t'], history['T_x0'])
plt.xlabel("t")
plt.ylabel("T_x0")
plt.title("Temperature evolution")
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
