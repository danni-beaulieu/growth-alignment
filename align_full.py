import numpy as np
from scipy.ndimage import shift
from scipy.optimize import minimize
from itertools import combinations

from losses import ncc_loss


# ----------------------------------------------------------
#  Objective: pairwise over all shifted images
# ----------------------------------------------------------
def objective(params, images):
    """
    params: [dx1, dy1, dx2, dy2, ..., dxN, dyN]
    images: list/array of grayscale images (float recommended)
    """
    N = len(images)
    shifts = np.array(params).reshape(N, 2)

    # Subpixel-shift each image
    shifted = [
        shift(img.astype(float),
              shift=(s[1], s[0]),  # (dy, dx)
              order=1,  # bilinear interpolation
              mode="constant",
              cval=0.0)
        for img, s in zip(images, shifts)
    ]

    total_loss = 0.0
    for i, j in combinations(range(N), 2):
        total_loss += ncc_loss(shifted[i], shifted[j])

    return total_loss


# ----------------------------------------------------------
#  Alignment wrapper
# ----------------------------------------------------------
def align_images(images, initial_guess=None):
    """
    images: list or array of grayscale images, shape [N, H, W]
    initial_guess: optional, shape (2N,)
    """
    N = len(images)

    if initial_guess is None:
        initial_guess = np.zeros(2 * N)

    result = minimize(
        objective,
        initial_guess,
        args=(images,),
        method="Powell",    # "L-BFGS-B" or "Powell"
        options={"maxiter": 250}
    )

    shifts = result.x.reshape(N, 2)
    return shifts


# ----------------------------------------------------------
# Example usage
# ----------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic grayscale images
    base = np.zeros((80, 80), dtype=float)
    base[20:60, 20:60] = 0.6  # square
    base[30:50, 30:50] = 1.0  # bright center

    img1 = base
    img2 = shift(base, shift=(5, 7))  # dy=5, dx=7
    img3 = shift(base, shift=(-3, 10))

    images = [img1, img2, img3]

    shifts = align_images(images)
    print("Estimated shifts (dx, dy):")
    print(shifts)
