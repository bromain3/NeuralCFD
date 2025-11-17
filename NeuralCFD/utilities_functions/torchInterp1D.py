import torch

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
    return (y0 + slope * (x - x0))