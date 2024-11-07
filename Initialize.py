# -*- coding: utf-8 -*-
"""
Initialization
"""
import numpy as np
from Constants import*
from funcs import *

def StandardInit(dx,days,inittype=0):
    '''
    Initialises parameters- dx, dt,nt,m,n and initialises arrays
    Parameters
    ----------
    dx : float
        spatial resolution in km.
    days : int
        Number of days to run the model for.
    inittype : int, optional
        Defines type of grid to be initialised. The default is 0.
        inittype=0: Arakawa C grid
        inittype=0: m x n analytical grid
    Returns
    -------
    u_initial : 2d array
        Array for zonal velocity with initial conditions.
    v_initial : 2d array
        Array for meridional velocity with initial conditions.
    eta_initial : 2d array
        Array for height perturbation with initial conditions.
    nt : int
        Number of time steps.
    m : int
        Maximum value of zonal index.
    n : int
        Maximum value of meridional index.
    dx : float
        spatial resolution in m.
    dt : float
        time step in seconds.

    '''
    dx=dx*1000    # convert km to m
    dt= (0.707*dx/100)*0.8
    endTime= days*3600*24
    nt=int(endTime/dt)
    
    m= int(L/dx)  # no of cells in x direction
    n=int(L/dx)  # no of cells in y direction
    
    #%% Grid Intialization
    if inittype==0:                      # arakawa C
        u_initial=np.zeros((n,m+1))
        v_initial=np.zeros((n+1,m)) 
        eta_initial=np.zeros((n,m))
        
    if inittype==1:                     # analytical
        u_initial=np.zeros((n+1,m+1))
        v_initial=np.zeros((n+1,m+1))
        eta_initial=np.zeros((n+1,m+1))
            
    return u_initial, v_initial, eta_initial,nt,m,n,dx,dt
