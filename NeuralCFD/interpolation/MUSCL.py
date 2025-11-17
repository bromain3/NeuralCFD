import torch

class MUSCL:
    def __init__(self, limiter):
        self.limiter = limiter

    def interpolate(self, leftInterfaces, rightInterfaces, cells):
        """
         Remplit leftInterfaces.Q[:, 1:ncells] et rightInterfaces.Q[:, 1:ncells]
         en respectant la convention du FirstOrderInterpolation donné.
         """
        Q = cells.Q  # shape (nvars, ncells)
        ncells = int(cells.ncells)

        nvars = Q.shape[0]

        # calcul des différences (d_minus et d_plus) pour chaque cellule
        # on veut d_minus[i] = Q[i] - Q[i-1], d_plus[i] = Q[i+1] - Q[i]
        # on construit arrays de taille (nvars, ncells)
        #TODO: finde a way to not create those arrays as it is time consuming
        d_minus = torch.zeros((nvars, ncells), device=cells.device, dtype=cells.dtype)
        d_plus = torch.zeros((nvars, ncells), device=cells.device, dtype=cells.dtype)

        # interior differences
        d_minus[:, 1:] = Q[:, 1:] - Q[:, :-1]  # for i>=1
        d_plus[:, :-1] = Q[:, 1:] - Q[:, :-1]  # for i<=ncells-2

        # For boundaries, d_minus[:,0] stays 0, d_plus[:,-1] stays 0.
        # A common approach is to set one-sided diffs equal to nearest available diff:
        # set d_minus[:,0] = d_plus[:,0] and d_plus[:,-1] = d_minus[:,-1]
        d_minus[:, 0] = d_plus[:, 0]
        d_plus[:, -1] = d_minus[:, -1]

        # apply limiter cell-wise: slope_i = phi(d_minus_i, d_plus_i)
        slope = self.limiter(d_minus, d_plus)  # shape (nvars, ncells)

        # Reconstruction at interfaces j = 1 .. ncells-1 (Python slice 1:ncells)
        # leftInterfaces.Q[:, j]  = Q[:, j-1] + 0.5 * slope[:, j-1]
        # rightInterfaces.Q[:, j] = Q[:, j]   - 0.5 * slope[:, j]

        # Initialize interface arrays unchanged (keep other indices as-is)
        # Only fill internal interfaces consistent with FirstOrderInterpolation
        leftInterfaces.Q[:, 1:ncells] = Q[:, 0:ncells - 1] + 0.5 * slope[:, 0:ncells - 1]
        rightInterfaces.Q[:, 1:ncells] = Q[:, 1:ncells] - 0.5 * slope[:, 1:ncells]

        # Note: indices written are
        # - leftInterfaces.Q[:, 1] uses slope[:, 0] and Q[:,0]
        # - rightInterfaces.Q[:, ncells-1] uses slope[:, ncells-1] and Q[:, ncells-1]
        # This exactly matches the convention used par FirstOrderInterpolation.

        return



def minmod(a, b):
    same_sign = (a * b > 0)
    return torch.where(same_sign, torch.where(torch.abs(a) < torch.abs(b), a, b), torch.zeros_like(a))

def vanleer(a, b):
    same_sign = (a * b > 0)
    return torch.where(same_sign, 2*a*b / (a + b), torch.zeros_like(a))

def superbee(a, b):
    return torch.maximum(
        torch.zeros_like(a),
        torch.maximum(torch.minimum(2*a, b), torch.minimum(a, 2*b))
    )

def vanalbada(a, b):
    same_sign = (a * b > 0)
    return torch.where(same_sign, (2*a*b) / (a*a + b*b + 1e-12), torch.zeros_like(a))

def mc(a, b):
    """Monotonized centered limiter (MC / van Leer MC)"""
    return torch.maximum(
        torch.zeros_like(a),
        torch.minimum(
            torch.minimum(0.5*(a + b), 2*a),
            2*b
        )
    )

def koren(a, b):
    """Koren limiter, good for smooth flows"""
    phi1 = torch.zeros_like(a)
    phi2 = torch.minimum(2*a, (2/3)*(a + b))
    phi3 = torch.minimum(a, 2*b)
    return torch.maximum(phi1, torch.minimum(phi2, phi3))

def monotonized_minmod(a, b):
    """Sometimes called minmod-Theta with theta=2"""
    theta = 2.0
    return minmod(a, theta * b)