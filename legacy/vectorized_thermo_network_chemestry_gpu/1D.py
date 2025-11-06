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

import matplotlib.pyplot as plt
import time
import torch

import thermo_lib as tl
import POLENE.PYTHONexport as PYTHONexport

# TODO: recode with shape of U being the same as the shape of Ys so that we can fully vectorize without transposing
#TODO: improve computations by computing all fields once befor each splitting operation so it is more computationally efficient even if the computations are quite quick after the first one
# or by not computing all fields whene one is enough

# ---------------------- Physical / numerical parameters ----------------------
# numerical parameters:
device = 'cpu'
dtype = torch.float64
torch.set_grad_enabled(False)

# Domain
L = 1.0
ncells = 200
dx = L / ncells
x = (torch.arange(ncells, device=device, dtype=dtype) + 0.5) * dx

# Thermodynamics params
y = tl.yamlParser(r'gri30.yaml')
vec_th = y.populateVectorizedThermoObject(dtype=dtype, device=device)
nspecies = vec_th.nspecies         # number of mass-fraction species
species_mw = vec_th.molar_masses  # shape (nspecies,), kg/mol

# chemestry model loading
cols, models, clusterer = PYTHONexport.loadPYTHONexportFolder(r'chem_models/export_PYTHON_const_pressure.pln')  #TODO: changing type of CFD chemical model when changing network model
tidy_cols = list(cols)
tidy_cols = [e for e in tidy_cols if e not in ('T', 'P')]
#TODO: make another check somewhere to be sure to give T in the good model'db_gen input and same for P
if not vec_th.species == tidy_cols: raise ValueError('model inputs and thermo not coherent')
# changing dtype of the model to double precision for the moment as it is what numpy uses
for i in models:
    i.model.to(dtype)
    i.model.eval()
    i.to(device)

# Numerical parameters
CFL = 0.9
final_time = 0.0005  # seconds (short demo)
max_steps = 10000



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
    T_int = torch.zeros(N+1, device=device, dtype=dtype)
    # interfaces internes : moyenne des cellules voisines
    T_int[1:N] = 0.5 * (T_prev[:-1] + T_prev[1:])
    # bords : extrapolation simple
    T_int[0] = T_prev[0]
    T_int[-1] = T_prev[-1]
    return T_int

def expand_T_guess_linear(T_prev):
    N = T_prev.shape[0]
    T_int = torch.zeros(N+1, device=device, dtype=dtype)
    # extrapolation à gauche et droite
    T_int[0] = 2*T_prev[0] - T_prev[1]
    T_int[-1] = 2*T_prev[-1] - T_prev[-2]
    # moyennes internes
    T_int[1:N] = 0.5 * (T_prev[:-1] + T_prev[1:])
    return T_int

def build_Ttab(Tmin=200.0, Tmax=6000.0, Ntab=300):
    """Créer la grille de températures pour la table (1D)."""
    return torch.linspace(Tmin, Tmax, Ntab, device=device, dtype=dtype)


def compute_etab(Ttab, Y, vec_th):
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
    T_flat = torch.tile(Ttab, (N,))              # (N*M,)
    # Y répété M fois par cellule
    Y_rep = torch.repeat_interleave(Y, M, dim=0)       # (N*M, nspecies)

    # h_mix vectorisé
    h_flat = vec_th.h_mix(T_flat, Y_rep)  # (N*M,)
    h_mat = h_flat.reshape(N, M)

    Rspec = vec_th.R_spec_from_Y(Y)[:, None]  # (N,1)
    etab = h_mat - Rspec * Ttab[None, :]      # (N,M)
    return etab

def torch_interp(x, xp, fp):
    """
    Interpolation linéaire 1D torch (équivalent à np.interp).
    x : scalaire ou tensor de valeurs à interpoler
    xp : tensor 1D (monotone croissant)
    fp : tensor 1D des valeurs correspondantes
    """
    # db_gen'assurer que xp est croissant
    indices = torch.searchsorted(xp, x.clamp(xp[0], xp[-1]))  # indices des intervalles
    indices = indices.clamp(1, len(xp)-1)

    x0, x1 = xp[indices-1], xp[indices]
    y0, y1 = fp[indices-1], fp[indices]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)

def guess_T_from_table(e, Y, Tmin=200.0, Tmax=6000.0, Ntab=300):
    """
    Estimer T à partir de e et Y en interpolant sur une table.
    e: (N,)
    Y: (N, nspecies)
    vec_th: ThermoData_vectorized
    Retour: T_guess (N,)
    """
    N = e.shape[0]
    Ttab = build_Ttab(Tmin, Tmax, Ntab)  # (M,)
    etab = compute_etab(Ttab, Y, vec_th) # (N,M)

    T_guess = torch.empty(N, dtype=dtype, device=device)

    # interpolation ligne par ligne (un petit for sur N)
    for i in range(N):
        et = etab[i]
        ei = e[i]

        if et[0] <= et[-1]:
            T_guess[i] = torch_interp(ei, et, Ttab)
        else:
            T_guess[i] = torch_interp(ei, torch.flip(et, dims=[0]), torch.flip(Ttab, dims=[0]))

    return T_guess

def solve_T_from_e_rho(e, rho, Y, T_guess=None, maxiter=50, tol=1e-6, max_T_step=0.2, err_type='absolute'):
    Rspec = vec_th.R_spec_from_Y(Y)
    #T_guess = None
    if T_guess is None:
        T = guess_T_from_table(e, Y)
    else:
        T = torch.clip(T_guess.clone(), 200.0, 6000.0)
    print('T_solver:',T[0])

    for i in range(maxiter):
        h = vec_th.h_mix(T, Y)
        f = h - Rspec*T - e
        cp = vec_th.cp_mix(T, Y)
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
    a = torch.full((N,), Tmin, device=device, dtype=dtype)
    b = torch.full((N,), Tmax, device=device, dtype=dtype)

    fa = vec_th.h_mix(a, Y) - Rspec * a - e
    fb = vec_th.h_mix(b, Y) - Rspec * b - e

    # Si fa et fb ont le même signe → pas de racine trouvée
    mask_fail = (fa * fb > 0)

    mid = torch.empty_like(e)
    for i in range(maxiter):
        mid = 0.5 * (a + b)
        fm = vec_th.h_mix(mid, Y) - Rspec * mid - e

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

    return T_out



# ---------------------- Utility converters -----------------------------------
def conserved_to_primitive(U, T_prev=None):
    """
    U shape: (3+nspecies, ncells)
    Returns:
      rho (ncells,), u (ncells,), p (ncells,), T (ncells,),
      Ys (ncells, nspecies), cv (ncells,), gamma (ncells,)
    """
    nvars, N = U.shape
    rho = U[0, :].clone()
    inv_rho = 1.0 / rho
    u = (U[1, :] * inv_rho).clone()
    E = (U[2, :] * inv_rho).clone()  # total energy per mass
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
    a = torch.sqrt(torch.clamp(gamma * Rspec * T, min=0.0))

    return rho, u, p, T, Ys, cv, gamma, e, a

def primitive_to_conserved(rho, u, p, T, Ys):
    """
    rho (ncells,), u (ncells,), p (ncells,), T (ncells,), Ys (ncells, nspecies)
    returns U shape (3+nspecies, ncells)
    """
    N = rho.size(dim=0)

    # compute internal energy per mass directly: e = h - R_spec * T
    Rspec = vec_th.R_spec_from_Y(Ys)
    h = vec_th.h_mix(T, Ys)
    e = h - Rspec * T

    kinetic = 0.5 * u**2
    E = e + kinetic  # total energy per mass

    U = torch.empty((3 + nspecies, N), device=device, dtype=dtype)
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
    F = torch.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = (rho * (e + 0.5 * u**2) + p) * u  # rho E * u + p u
    F[3:3+nspecies, :] = Ys.T * rho * u
    return F

def max_wave_speed(U, Tprev=None):
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)
    return torch.max(torch.abs(u) + a)

def max_wave_speed_vectorized(U, Tprev=None):
    rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)
    return torch.abs(u) + a


def rusanov_flux_vectorized(UL, UR, Tprev=None):
    # UL, UR are states at left and right of interface (shape: nvars)
    # We will vectorize this in the transport step.
    FL = flux(UL, Tprev)
    FR = flux(UR, Tprev)
    sL = max_wave_speed_vectorized(UL, Tprev)
    sR = max_wave_speed_vectorized(UR, Tprev)
    smax = torch.maximum(sL, sR)
    # Rusanov: F = 0.5*(FL+FR) - 0.5*smax*(UR-UL)
    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL), torch.max(smax)

# ---------------------- Transport step (finite volume) ----------------------
def transport_step(U, dt, Tprev=None, bc='periodic', inflow_state=None):
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
    UL_int = torch.zeros((nvars, N + 1), device=device, dtype=dtype)
    UR_int = torch.zeros((nvars, N + 1), device=device, dtype=dtype)
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
    F_int, smax_global = rusanov_flux_vectorized(UL_int, UR_int, Tprev)


    # update conserved: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    U_new = U.clone()
    U_new[:, :] = U[:, :] - (dt / dx) * (F_int[:, 1:] - F_int[:, :-1])
    return U_new, smax_global

# ---------------------- Chemistry step (calls user'db_gen NN) ----------------------
def chemistry_step_with_nn(U, dt, nn_predict, Tprev=None, reactor_model='cst_p_const_rhou'):
    if reactor_model == 'cst_v':
        # chemistry acts locally cell-by-cell. We keep rho and rho*u constant, update E and Ys
        rho = U[0].clone()
        u = U[1] / rho
        _, _, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)  # Ys shape (ncells, nspecies)
        # Call the NN: it should accept arrays and return arrays
        T_new, p_new, Ys_new = nn_predict(T, p, Ys, dt)
        # Sanity shapes
        assert T_new.shape == T.shape
        assert p_new.shape == p.shape
        assert Ys_new.shape == Ys.shape
        # Reconstruct conserved: keep rho and rho*u, update rho*E and rho*Y. it means no mass transfer between cells during chemestry, so constant volume
        # TODO : for consistency, we can recompute P however for the moment, the least corrections we do to the nn predictions, the least likely it is for it to do wierd shit
        #Rspec_new = vec_th.R_spec_from_Y(Ys_new)
        #p_new = rho * Rspec_new * T_new

    elif reactor_model == 'cst_p_const_u':
        # chemistry acts locally cell-by-cell. We keep P and u constant, update rho, E and Ys
        rho, u, p, T, Ys, cv, gamma, e, a = conserved_to_primitive(U, Tprev)  # Ys shape (ncells, nspecies)
        # Call the NN: it should accept arrays and return arrays
        T_new, p_new, Ys_new = nn_predict(T, p, Ys, dt)
        p_new = p
        # Sanity shapes
        assert T_new.shape == T.shape
        assert p_new.shape == p.shape
        assert Ys_new.shape == Ys.shape
        # Reconstruct conserved: keep p and u, update rho, rho*E and rho*Y
        Rspec_new = vec_th.R_spec_from_Y(Ys_new)
        rho = p_new / (Rspec_new * T_new)

    elif reactor_model == 'cst_p_const_rhou': # most physical and coherent with numerical spliting
        # chemistry acts locally cell-by-cell. We keep P and u constant, update rho, E and Ys
        rho_old = U[0, :].clone()
        mom_old = U[1, :].clone()  # rho * u
        u_old = mom_old / rho_old
        _, _, p_old, T_old, Ys_old, _, _, e_old, _ = conserved_to_primitive(U, Tprev)  # Ys shape (ncells, nspecies)
        # Call the NN: it should accept arrays and return arrays
        T_new, p_new, Ys_new = nn_predict(T_old, p_old, Ys_old, dt)
        p_new = p_old
        # Sanity shapes
        assert T_new.shape == T_old.shape
        assert p_new.shape == p_old.shape
        assert Ys_new.shape == Ys_old.shape
        # Reconstruct conserved: keep p and u, update rho, rho*E and rho*Y
        Rspec_new = vec_th.R_spec_from_Y(Ys_new)
        rho = p_new / (Rspec_new * T_new)
        u = mom_old / rho

    else:
        raise ValueError('Unknown reactor model')

    U_new = primitive_to_conserved(rho, u, p_new, T_new, Ys_new)
    return U_new

# ---------------------- Dummy NN / user should replace ------------------------
def dummy_nn_predict(T, p, Ys, dt): #TODO: optimize memory usage
    # T, p: shape (ncells,), Ys: shape (ncells, nspecies)
    input = torch.hstack((T[:, None], p[:, None], Ys))
    clust = torch.tensor(clusterer.computeClusterIndexTorch(input), device=device, dtype=dtype)

    print('nn_input:', T)

    # 1. trier les indices par cluster
    sorted_idx = torch.argsort(clust)
    sorted_clust = clust[sorted_idx]

    results = torch.zeros_like(input)

    # 2. trouver les frontières de changement de cluster
    boundaries = torch.where(sorted_clust[1:] != sorted_clust[:-1])[0] + 1
    splits = torch.tensor_split(sorted_idx, boundaries.tolist())
    # 3. boucle cluster par cluster
    for idx_block in splits:
        cid = int(clust[idx_block[0]].item())  # cluster id pour ce bloc
        X_cluster = input[idx_block]
        y_pred = models[cid].inputScaler.transformTorch(X_cluster)
        y_pred = models[cid](y_pred)
        y_pred = models[cid].outputScaler.inverse_transformTorch(y_pred)

        results[idx_block] = y_pred

    T_new = results[:, 0].flatten()
    p_new = results[:, 1].flatten()
    Ys_new = results[:, 2:]
    print('nn_output:', T_new)
    return T_new, p_new, Ys_new

# ---------------------- Initial condition -----------------------------------
def initial_condition():
    #rho0 = 1.0 + 0.0 * x
    p0 = 101325.0 + 0.0 * x
    u0 = 0.0 + 0.0 * x
    T0 = 1700.0 + 0.0 * x #1650.0 + 100.0 * torch.exp(-((x - 0.5)**2) / (2 * 0.02**2))  # hot spot in center #1700.0 + 0.0 * x
    Ys0 = torch.zeros((ncells, nspecies), device=device, dtype=dtype)
    #Ys0[:, 0] = 0.5 * np.exp(-((x - 0.5)**2) / (2 * 0.05**2)) + 0.25
    #Ys0[:, 1] = 1.0 - Ys0[:, 0]
    Ys0[:, vec_th.species.index('CH4')] = 0.05483547715890419 + 0.0 * x
    Ys0[:, vec_th.species.index('N2')] = 0.7263764388501015 + 0.0 * x
    Ys0[:, vec_th.species.index('O2')] = 0.21878808399099442 + 0.0 * x

    Rspec0 = vec_th.R_spec_from_Y(Ys0)
    #p0 = rho0 * Rspec0 * T0
    rho0 = p0 / Rspec0 / T0

    U0 = primitive_to_conserved(rho0, u0, p0, T0, Ys0)
    print(T0[0], p0[0], Ys0[0])
    return U0

# ---------------------- Time integration with Strang splitting ----------------
def run_solver(nn_predict, final_time, U0):
    U = U0.clone()
    t = 0.0
    step = 0
    history = {'t': [], 'T': [], 'T_x0': [], 'Y0': [], 'rhoE':[]}

    _, _, _, T_prev, _, _, _, _, _ = conserved_to_primitive(U)

    while t < final_time and step < max_steps:
        # compute stable dt by CFL
        #smax = torch.max(max_wave_speed_vectorized(U)).item()
        #dt_cfl = CFL * dx / (smax + 1e-12)
        #dt = min(dt_cfl, final_time - t)
        dt = 1e-6


        # Strang splitting: half transport, full chemistry, half transport
        U_half, s1 = transport_step(U, 0.5 * dt, T_prev, bc='periodic', inflow_state=inflow_state)
        U_chemed = chemistry_step_with_nn(U_half, dt, nn_predict, T_prev)
        U_new, s2 = transport_step(U_chemed, 0.5 * dt, T_prev, bc='periodic', inflow_state=inflow_state)

        U = U_new
        t += dt
        step += 1

        rho, u, p, T_prev, Ys, _, _, _, _ = conserved_to_primitive(U, T_prev)

        if step % 1 == 0 or t >= final_time:
            history['t'].append(t)
            history['T'].append(T_prev.detach().clone().cpu().numpy())
            history['T_x0'].append(T_prev.detach().clone().cpu().numpy()[0])
            history['Y0'].append(Ys[:, vec_th.species.index('CH4')].detach().clone().cpu().numpy())
            history['rhoE'].append(torch.sum(U[2]).detach().clone().cpu().numpy())
    return U, T_prev, history, step

# ---------------------- Run demo --------------------------------------------
p0 = torch.tensor([101325.0], dtype=dtype, device=device)  # conditions d’entrée
u0 = torch.tensor([1.0], dtype=dtype, device=device)
T0 = torch.tensor([1650.0], dtype=dtype, device=device)
Ys0 = torch.zeros((1, nspecies), dtype=dtype, device=device)
Ys0[:, vec_th.species.index('CH4')] = 0.05483547715890419
Ys0[:, vec_th.species.index('N2')] = 0.7263764388501015
Ys0[:, vec_th.species.index('O2')] = 0.21878808399099442


# compute p0 vectorisé
Rspec_inflow = vec_th.R_spec_from_Y(Ys0)
#p0 = rho0 * Rspec_inflow * T0
rho0 = p0 / Rspec_inflow / T0


inflow_state = primitive_to_conserved(rho0, u0, p0, T0, Ys0).flatten()

U0 = initial_condition()
t0 = time.time()
U_final, T_prev, history, steps = run_solver(dummy_nn_predict, final_time, U0)

print('simulation took %.10f seconds' % (time.time() - t0))
print(f"Finished in {steps} steps, final time approx {history['t'][-1]:.6e} s")

rho, u, p, T_final, Ys_final, _, _, _, _ = conserved_to_primitive(U_final, T_prev)
rho = rho.clone().detach().cpu().numpy()
u = u.clone().detach().cpu().numpy()
p = p.clone().detach().cpu().numpy()
T_final = T_final.clone().detach().cpu().numpy()
Ys_final = Ys_final.clone().detach().cpu().numpy()

x = x.clone().detach().cpu().numpy()

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
