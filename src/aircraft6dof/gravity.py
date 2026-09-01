import numpy as np

def normal_gravity(latitude_rad,altitude_m):
    s=np.sin(latitude_rad)
    g0=9.7803253359*(1+0.00193185265241*s*s)/np.sqrt(1-0.00669437999013*s*s)
    return g0-3.086e-6*altitude_m

def gravity_ned(latitude_rad,altitude_m):
    return np.array([0.,0.,normal_gravity(latitude_rad,altitude_m)])
