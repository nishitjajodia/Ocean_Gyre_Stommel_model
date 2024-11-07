# -*- coding: utf-8 -*-
"""
functions for the report
"""

import numpy as np
from Constants import*
from funcs import *
import TimeSchemes as TS
import Initialize as Init
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import time
#%%
def windstress():
    '''
    Plots spatial distribution of wind stress vector

    Returns
    -------
    None.

    '''
    x = np.linspace(0, 10**6, 10)
    y = np.linspace(0, 10**6, 10)
    X, Y = np.meshgrid(x, y)
    
    taux=-0.2*np.cos(np.pi*Y/L)
    tauy=0*X
    fig, ax = plt.subplots(1,2,figsize=(10,2))
    ax[0].quiver(X, Y, taux, tauy)
    ax[0].set_xlabel('x (m)')
    ax[0].set_ylabel('y (m)')
    ax[0].set_title('Wind stress vector field plot')
    
    ax[1].plot(taux[:,-1],y)
    ax[1].set_xlabel('$tau_x$')
    ax[1].set_ylabel('y (m)')
    ax[1].set_title('Wind stress')
    ax[1].grid()
    
    plt.show()
       
def TaskC(dx,title,gridtype=1):
    '''
    Runs code for Task C

    Parameters
    ----------
    dx : float
        spatial resolution in km.
    title : string
        title of plot.
    gridtype : int, optional
        Defines type of grid to be initialised.
        inittype=0: Arakawa C grid
        inittype=0: m x n analytical grid The default is 1.

    Returns
    -------
    u_st : 2d array
        analytical u at steady state.
    v_st :  2d array
        analytical v at steady state.
    eta_st : 2d array
        analytical $\eta$ at steady state.
    energyAn : float
        Total energy of system at steady state.

    '''
    u_st,v_st,eta_st,energyAn=TS.analytical(dx,gridtype)
    contplot(u_st,v_st,eta_st,title)
    
    return u_st,v_st,eta_st,energyAn

def TaskD12(dx,days):
    '''
    Runs Tasks D.1 and D.2

    Parameters
    ----------
    dx :  float
        spatial resolution in km.
    days : int
        Number of days to run the model for.

    Returns
    -------
    u_u : 2d array
        numerical solution of u .
    v_v : 2d array
        numerical solution of v.
    eta : 2d array
        numerical solution of eta.
    dx : float
        spatial resolution in m.
    dt :float
        time step in seconds.
    nt : int
        Number of time steps.
    energyTS : 1d array
        Total energy time series of system.

    '''

    start_time=time.time()
    u_u,v_v,eta,dx,dt,nt,energyTS=TS.FBM(dx,days,0,True) # FB-1 day
    FBtime = time.time() - start_time
    print('Time to run model= ',FBtime,' seconds')
    contplot(u_u,v_v,eta,'Forward-Backward Scheme')
    
    plt.figure(figsize=(15,6))
    
    plt.subplot(2,3,1)
    plotx(u_u,dx,2,'u (m/s)')
    plotx(u_u,dx,0,'u (m/s)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.title('u vs x')
    plt.legend()
    plt.grid()
    
    plt.subplot(2,3,2)
    ploty(v_v,dx,0,'v (m/s)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.title('v vs y')
    ploty(v_v,dx,1,'v (m/s)')
    plt.legend()
    plt.grid()
    
    plt.subplot(2,3,3)
    plotx(eta,dx,1,'$\eta (m)$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.title('$\eta$ vs x at centre of grid')
    
    return u_u,v_v,eta,dx,dt,nt,energyTS

def TaskD3(u_u,v_v,eta,u_st,v_st,eta_st,dx,title):
    '''
    Runs Task D.3
    
    Parameters
    ----------
    u_u : 2d array
        numerical solution of u .
    v_v : 2d array
        numerical solution of v.
    eta : 2d array
        numerical solution of eta.
    u_st : 2d array
        analytical u at steady state.
    v_st :  2d array
        analytical v at steady state.
    eta_st : 2d array
        analytical $\eta$ at steady state.
    dx : float
        spatial resolution in m.
    title : tring
        title of figure

    Returns
    -------
    None.

    '''
    
    u1=u_u-u_st
    v1= v_v-v_st
    eta1=eta-eta_st # differences
    contplot(u1,v1,eta1,title,True)                                   # contour plots for differences
    
    DeltaE="{:e}".format(Energy(u1,v1,eta1,dx))
    print('At steady state, the total energy difference between Analytical and numerical methods is ',DeltaE,' Joules')


def TaskE(u_u,v_v,eta,nt,dx,dt,energyTS):
    '''
    Runs Task E

    Parameters
    ----------
    u_u : 2d array
        numerical solution of u .
    v_v : 2d array
        numerical solution of v.
    eta : 2d array
        numerical solution of eta.
    dx : float
        spatial resolution in m.
    dt :float
        time step in seconds.
    nt : int
        Number of time steps.
    energyTS : 1d array
        Total energy time series of system.

    Returns
    -------
    None.

    '''

    fig, ax = plt.subplots(1,3,figsize=(15,3))
    timeticks=np.linspace(0,40,nt)
    ax[0].plot(timeticks,energyTS) 
    ax[0].grid()
    ax[0].set_title('Total energy vs Time')
    ax[0].set_ylabel('Total energy (J)')
    ax[0].set_xlabel('dx (km)')
    steadytime=steadyfinder(energyTS,dt)
    
    # Delta E vs dx plot
    nd=6
    energydiff=np.zeros(nd)
    error_percent=np.zeros(nd)
    dxvary=np.linspace(10,30,nd)
    for i in range(nd):      
        u_st,v_st,eta_st,energyAn=TS.analytical(dxvary[i],1)  # analytical
        u_u,v_v,eta,dx1,dt,nt,energyTS=TS.FBM(dxvary[i],40,0,True)   # forward-backward
        u1,v1,eta1=u_u-u_st, v_v-v_st, eta-eta_st #differences
        energydiff[i]=Energy(u1,v1,eta1,dx1) # energy difference
        error_percent[i]=(energydiff[i]/energyAn)*100
    ax[1].plot(dxvary,energydiff)
    ax[1].scatter(dxvary,energydiff)
    ax[1].set_title('$\Delta$ E vs dx')
    ax[1].set_ylabel('$\Delta E (J)$')
    ax[1].set_xlabel('dx (km)')
    ax[1].grid()
    ax[2].plot(dxvary,error_percent)
    ax[2].scatter(dxvary,error_percent)
    ax[2].set_title('Energy Error % vs dx')
    ax[2].set_ylabel('Energy Error %')
    ax[2].set_xlabel('dx (km)')
    ax[2].grid()
    plt.show()


def TaskG4(u_st,v_st,eta_st,dx,days):
    '''
    Runs Task G.4: 4th order Runge-Kutta Scheme

    Parameters
    ----------
    u_st : 2d array
        analytical u at steady state.
    v_st :  2d array
        analytical v at steady state.
    eta_st : 2d array
        analytical $\eta$ at steady state.
    dx : float
        spatial resolution in km.
    days : int
        number of days to run the models for.

    Returns
    -------
    None.

    '''
    start_time=time.time()
    u_u,v_v,eta,dx,dt,nt,energyTS=TS.RK4(dx,days) # FB-1 day
    RKtime = time.time() - start_time
    print('Time to run model= ',RKtime,' seconds')
    contplot(u_u,v_v,eta,'Runge-Kutta Scheme at the end of 40 days')
    
    plt.figure(figsize=(15,6))
    
    plt.subplot(2,3,1)
    plotx(u_u,dx,2,'u (m/s)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.title('u vs x at southern edge')
    
    plt.subplot(2,3,2)
    ploty(v_v,dx,0,'v (m/s)')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.title('v vs y at western edge')
    
    plt.subplot(2,3,3)
    plotx(eta,dx,1,'$\eta (m)$')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.title('$\eta$ vs x at centre of grid')
    
    u1=u_u-u_st
    v1= v_v-v_st
    eta1=eta-eta_st # differences
    contplot(u1,v1,eta1,'Difference between Analytical and RK4 Numerical Solutions',True)