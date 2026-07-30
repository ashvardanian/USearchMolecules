import numpy as np


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    a = np.unpackbits(a.view(np.uint8))
    b = np.unpackbits(b.view(np.uint8))
    ands = np.logical_and(a, b).sum()
    ors = np.logical_or(a, b).sum()
    return 1 - ands / ors
