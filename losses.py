import numpy as np


# ----------------------------------------------------------
#  1 - Overlap
# ----------------------------------------------------------
def overlap_loss(a, b):
    bitwiseand = np.bitwise_and(a, b)
    total = a + b
    total_pixels = total[total > 0].shape[0]
    matches = bitwiseand[bitwiseand > 0].shape[0]
    return 1 - (matches / total_pixels)


# ----------------------------------------------------------
#  XOR
# ----------------------------------------------------------
def xor_loss(a, b):
    xor_difference = np.bitwise_xor(a, b)
    total = a + b
    total_pixels = total[total > 0].shape[0]
    differences = xor_difference[xor_difference > 0].shape[0]
    return differences / total_pixels


# ----------------------------------------------------------
#  Continuous Dice
# ----------------------------------------------------------
def soft_dice_loss(a, b, eps=1e-6):
    """
    Continuous Dice loss suitable for floating masks.
    """
    inter = np.sum(a * b)
    sa = np.sum(a)
    sb = np.sum(b)
    dice = (2 * inter + eps) / (sa + sb + eps)
    return 1.0 - dice


# ----------------------------------------------------------
#  Normalized Cross-Correlation
# ----------------------------------------------------------
def ncc(a, b, eps=1e-8):
    a_mean = a - np.mean(a)
    b_mean = b - np.mean(b)
    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean**2) * np.sum(b_mean**2)) + eps
    return num / den


def ncc_loss(a, b):
    return 1.0 - ncc(a, b)


# ----------------------------------------------------------
#  Normalized Gradient Fields
# ----------------------------------------------------------

def ngf_loss(a, b, eps=1e-6):
    ax = np.gradient(a, axis=1)
    ay = np.gradient(a, axis=0)
    bx = np.gradient(b, axis=1)
    by = np.gradient(b, axis=0)

    dot = ax * bx + ay * by
    na = ax**2 + ay**2 + eps
    nb = bx**2 + by**2 + eps

    ngf_val = np.mean(1.0 - (dot**2 / (na * nb)))
    return ngf_val


# ----------------------------------------------------------
#  Mututal information
# ----------------------------------------------------------

def mutual_information(a, b, bins=32):
    hist_2d, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz]))
    return mi

def mi_loss(a, b):
    return -mutual_information(a, b)  # maximize MI → minimize loss


# -------------------------
# L2 Loss
# -------------------------

def l2_mean(a, b):
    """Mean squared error (standard L2 loss)."""
    diff = a - b
    return np.mean(diff * diff)

def l2_sum(a, b):
    """Sum of squared differences (SSD)."""
    diff = a - b
    return np.sum(diff * diff)

def l2_norm(a, b):
    """Euclidean norm between images."""
    diff = a - b
    return np.sqrt(np.sum(diff * diff))

def huber(a, b, delta=0.01):
    d = np.abs(a - b)
    return np.where(d < delta, 0.5*d*d, delta*(d - 0.5*delta)).mean()


# -------------------------
# Weighted/Masked L2 Loss
# -------------------------

def weighted_l2(a, b, w):
    diff = a - b
    return np.mean(w * diff * diff)

def masked_l2(a, b, mask):
    """
    a, b: images (same shape)
    mask: boolean or 0/1 mask, same shape; 1 = valid pixel, 0 = ignore
    """
    diff = (a - b) * mask
    count = np.sum(mask)

    if count == 0:
        return 0.0  # no valid pixels → no contribution

    return np.sum(diff * diff) / count

