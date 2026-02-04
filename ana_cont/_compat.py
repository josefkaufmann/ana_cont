import numpy as np

try:
    trapz = np.trapezoid  # valid for numpy>=2.0.0
except AttributeError:
    trapz = np.trapz  # for older numpy versions