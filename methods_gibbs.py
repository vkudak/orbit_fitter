import numpy as np
from constants import MU_EARTH

def gibbs_velocity(r1, r2, r3):
    c12 = np.cross(r1, r2)
    c23 = np.cross(r2, r3)
    c31 = np.cross(r3, r1)
    N = np.linalg.norm(r1) * c23 + np.linalg.norm(r2) * c31 + np.linalg.norm(r3) * c12
    D = c12 + c23 + c31
    S = (np.linalg.norm(r2) - np.linalg.norm(r3)) * r1 + (np.linalg.norm(r3) - np.linalg.norm(r1)) * r2 + (np.linalg.norm(r1) - np.linalg.norm(r2)) * r3
    if np.linalg.norm(N) < 1e-12 or np.linalg.norm(D) < 1e-12:
        return (r3 - r1) / (np.linalg.norm(r3 - r1) + 1e-9)
    v2 = np.cross(D, r2) * (1.0 / (np.linalg.norm(r2)**2)) * np.sqrt(MU_EARTH / (np.linalg.norm(N) * np.linalg.norm(D))) + S * (1.0 / np.linalg.norm(N))
    return v2
