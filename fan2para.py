import math
import numpy as np








def fan2para(F, D):
    """
    :param F: fan beam data
    :param D: D is the distance in pixels from the fan-beam vertex to the center of rotation that was used to obtain the projections.
    :return:
    """


    gammaDeg = formGamaVector()



def formGamaVector():
    m = 128
    m2cn = math.floor((m - 1) / 2)
    m2cp = math.floor(m / 2)
    g = np.arange(-m2cn, m2cp + 1)
    return g