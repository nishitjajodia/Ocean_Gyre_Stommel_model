# -*- coding: utf-8 -*-
"""
Defines constants to be used in the project
"""
import numpy as np

g=9.81 #m/s2
pi=np.pi
gamma= 10**-6 # drag coefficient
rho=1000 # density in kg/m3
H=1000  # m, depth of ocean
L=10**6 #m, horizontal legth scale
tau0=0.2 #N/m2, tau0, for tau= wind stress
f0=10**-4 #s-1, constant coriolos parameter (f-plane)
Beta= 10**-11 # m-1s-1

E=gamma/(L*Beta)
a=(-1 - np.sqrt( 1+ (2*pi*E)**2 ) ) / (2*E)
b=(-1 + np.sqrt( 1+ (2*pi*E)**2 ) ) / (2*E)

eta0=-0.08878321179979096  # from FB result