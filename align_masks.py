import numpy as np
from scipy.ndimage import shift
from scipy.optimize import minimize
from itertools import combinations

from losses import ncc_loss

def objective(params, masks):
    """
    params = [x1, y1, x2, y2, ..., xN, yN]
    masks = list/array of binary masks with same shape
    """
    N = len(masks)
    shifts = np.array(params).reshape(N, 2)

    shifted = [shift(m, (s[1], s[0]))
               for m, s in zip(masks, shifts)]


    total_loss = 0.0
    for i, j in combinations(range(N), 2):
        total_loss += ncc_loss(shifted[i], shifted[j])

    return total_loss


def align_masks(masks, initial_guess=None):
    """
    masks: list/array of shape [N, H, W]
    initial_guess: None or array of shape (2N,)
    """
    N = len(masks)

    if initial_guess is None:
        initial_guess = np.zeros(2 * N)

    result = minimize(
        objective,
        initial_guess,
        args=(masks,),
        method="Powell"
    )

    shifts = result.x.reshape(N, 2)
    return shifts, result


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    # Create example masks
    mask1 = np.zeros((50, 50), dtype=np.uint8)
    mask1[10:30, 10:30] = 1

    mask2 = np.zeros((50, 50), dtype=np.uint8)
    mask2[15:35, 15:35] = 1

    mask3 = np.zeros((50, 50), dtype=np.uint8)
    mask3[5:25, 20:40] = 1

    masks = [mask1, mask2, mask3]
    shifts = align_masks(masks)
    print("Optimal shifts:", shifts)