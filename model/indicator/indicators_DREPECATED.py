from math import sqrt
import numpy as np

def elapsed_time(t_n, t_1):
    """
    The time elapsed over a predefined number of events (T).
    """
    return t_n - t_1


def mean_magnitude(m):
    """
    The mean magnitude of n events.
    """
    return sum(m) / len(m)


def seismic_energy(m):
    try:
        iter(m)
        return sum(10 ^ ((1.5 * m) + 4.8))
    except TypeError:
        return 10 ^ ((1.5 * m) + 4.8)


def rate_of_square_root_of_seismic_energy(m, n):
    """
    The rate of square root of seismic energy released over time 't'
    """

    m = np.array(m)
    n = np.array(n)
    n_items = len(m)

    part1 = np.sum(n * n)
    part2 = np.sum(m) * np.sum(n)
    part3 = np.square(np.sum(m))
    part4 = np.sum(np.square(m))
    if n == 1:
        b = 0.0
        a = np.sum(n) / n
    else:
        b = (n * part1 - part2) / (part3 - n * part4)
        M = b * np.array(m)
        a = np.sum(n + m) / n
    return 1


def b_value(m, n):
    """
    Slope of the log of the earthquake frequency versus magnitude curve.
    """

    n = len(m)
    return
