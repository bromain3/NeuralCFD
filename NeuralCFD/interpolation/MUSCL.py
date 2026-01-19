import torch

class MUSCL:
    def __init__(self, limiter):
        self.limiter = limiter

    def interpolate(self, leftInterfaces, rightInterfaces, cells):
        """
        Interpole les valeurs cell-centered vers les interfaces (MUSCL)
        leftInterfaces: objet avec Q[:, i] pour l'interface i
        rightInterfaces: idem
        cells: objet avec Q[:, i] pour les cellules
        """
        Q = cells.Q
        nVar, nCells = Q.shape

        # Calcul des pentes
        delta_left = Q[:, 1:-1] - Q[:, :-2]  # ΔQ_{i-1/2} = Q_i - Q_{i-1}
        delta_right = Q[:, 2:] - Q[:, 1:-1]  # ΔQ_{i+1/2} = Q_{i+1} - Q_i

        slope = self.limiter(delta_left, delta_right)  # pente limitée

        # Interpolation MUSCL (linéaire)
        rightInterfaces.Q[:, 1:-2] = Q[:, 1:-1] - 0.5 * slope
        leftInterfaces.Q[:, 2:-1] = Q[:, 1:-1] + 0.5 * slope

        # Gestion des bords (first-order fallback)
        leftInterfaces.Q[:, 1] = Q[:, 0]
        leftInterfaces.Q[:, -1] = Q[:, -1]
        rightInterfaces.Q[:, -2] = Q[:, -1]
        rightInterfaces.Q[:, 0] = Q[:, 0]



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

def ospre(a, b):
    """
    OSPRE flux limiter.
    a = du_{i}   (slope left)
    b = du_{i+1} (slope right)
    """
    eps = 1e-12
    same_sign = (a * b > 0)

    r = a / (b + eps)

    phi_r = 1.5 * r * (r + 1.0) / (r*r + r + 1.0 + eps)

    # slope limited value = φ(r) * b
    out = phi_r * b

    return torch.where(same_sign, out, torch.zeros_like(a))