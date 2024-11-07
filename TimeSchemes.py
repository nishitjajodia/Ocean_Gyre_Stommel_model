# -*- coding: utf-8 -*-
"""
time schemes
"""
import numpy as np
from Constants import*
from funcs import *
import Initialize as Init
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

# start_time=time.time()

def analytical(dx,gridtype=1):
    '''
    Generates analytical solution for given saptial resolution, dx
    Parameters
    ----------
    dx : float
        spatial resolution in km.
    gridtype :int, optional
            Defines type of grid to be initialised.
            inittype=0: Arakawa C grid
            inittype=0: m x n analytical grid. The default is 1.

    Returns
    -------
    ust1 : 2d array
        analytical u at steady state.
    vst1 : 2d array
        analytical v at steady state.
    etast1 : 2d array
        analytical $\eta$ at steady state.
    energyAn : float
        Total energy of system at steady state.

    '''
    u_st,v_st,eta_st,nt,m,n,dx,dt=Init.StandardInit(dx,0,gridtype)  # initialize grids
    for j in range(n):   
        for i in range(m):       
            u_st[j,i]= -(tau0/(pi*gamma*rho*H))*f1(i*dx/L)*np.cos(pi*(n-1-j)*dx/L)
            v_st[j,i]= (tau0/(pi*gamma*rho*H))*f2(i*dx/L)*np.sin(pi*(n-1-j)*dx/L)
            eta_st[j,i]= eta0 + (tau0/(pi*gamma*rho*H))*(f0*L/g)*( gamma/(f0*pi)*f2(i*dx/L)*np.cos(pi*(n-1-j)*dx/L) + (1/pi)*f1(i*dx/L)*( np.sin(pi*(n-1-j)*dx/L)*(1 + Beta*(n-1-j)*dx/f0) + (Beta*L/(f0*pi))*np.cos(pi*(n-1-j)*dx/L) ) )
    ust1,vst1,etast1=AnalyticalonArakawa(u_st,v_st,eta_st)
    energyAn=Energy(ust1,vst1,etast1,dx)
    return ust1,vst1,etast1,energyAn

#%%
def FBM(dx,days,Initcond=0,windON=True):
    '''
    Generates Forward-Backward solution for given saptial resolution, dx and nummber of days

    Parameters
    ----------
    dx : float
        spatial resolution in km.
    days : int
        Number of days to run the model for.
    Initcond : int, optional
            Defines type of grid to be initialised.
            inittype=0: Arakawa C grid
            inittype=0: m x n analytical grid. The default is 0.
    windON : optional
            The default is True.

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
        Total energy of system.

    '''
    if Initcond==0:
        u_u,v_v,eta,nt,m,n,dx,dt=Init.StandardInit(dx,days)   
    # wind toggle
    if windON==True:
        win=1
    else:
        win=0
    dy=dx
    
    energyTS=np.zeros(nt)
    # on u grid
    y_u=np.zeros((n,m+1))
    for i in range(m+1):
        y_u[:,i]=np.linspace(L-dy/2,dy/2,n)  # y on u grid
    fu=(f0+Beta*y_u[:,1:m])

    # on v grid
    y_v=np.zeros((n+1,m))
    for i in range(m):    
        y_v[:,i]=np.linspace(L,0,n+1)      # y on v grid
    fv=(f0+Beta*y_v[1:n,:])

    for k in range(1,nt):
        eta_last=eta
        u_last=u_u
        v_last=v_v
        eta=eta_last - H*dt*( dudx(dx,u_last) + dvdy(dy,v_last) )    
        
        if k%2==1:
            # interpolating v from previous time-step
            v_int=calc_Vint(v_last)
            u_u[:,1:m]= u_last[:,1:m] + dt*(fu*v_int - g*detadx(dx,eta) - gamma*u_last[:,1:m] + win*tau(y_u[:,1:m],L)[0]/(rho*H))
            
            #interpolating u from current time-step
            u_int=calc_Uint(u_u)
            v_v[1:n,:]= v_last[1:n,:] + dt*(- fv*u_int - g*detady(dy,eta) - gamma*v_last[1:n,:] + win*tau(y_v[1:n,:],L)[1]/(rho*H))
            
        if k%2==0:
            # interpolating u from previous time-step
            u_int=calc_Uint(u_last)
            v_v[1:n,:]= v_last[1:n,:] + dt*(- fv*u_int - g*detady(dy,eta) - gamma*v_last[1:n,:] + win*tau(y_v[1:n,:],L)[1]/(rho*H))
            
            #interpolating v from current time-step      
            v_int=calc_Vint(v_v)
            u_u[:,1:m]= u_last[:,1:m] + dt*(fu*v_int - g*detadx(dx,eta) - gamma*u_last[:,1:m] + win*tau(y_u[:,1:m],L)[0]/(rho*H))
        
        energyTS[k]=Energy(u_u,v_v,eta,dx)
        
    return u_u, v_v, eta,dx,dt,nt,energyTS

#%%

def RK4(dx,days):
    '''

    Parameters
    ----------
    dx : float
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
       Total energy of system.

    '''
    u_u,v_v,eta,nt,m,n,dx,dt=Init.StandardInit(dx,days)
    dy=dx
    energyTS=np.zeros(nt)
    # on u grid
    y_u=np.zeros((n,m+1))
    for i in range(m+1):
        y_u[:,i]=np.linspace(L-dy/2,dy/2,n)  # y on u grid
    fu=(f0+Beta*y_u[:,1:m])

    # on v grid
    y_v=np.zeros((n+1,m))
    for i in range(m):    
        y_v[:,i]=np.linspace(L,0,n+1)      # y on v grid
    fv=(f0+Beta*y_v[1:n,:])
    
    for j in range(1,nt):
        eta_last=eta
        u_last=u_u
        v_last=v_v
        etaP=eta_last.copy()
        uP=u_last.copy()
        vP=v_last.copy()
        
        K1=calc_K(u_last,v_last,dx)
        L1=calc_L(u_last,v_last,eta,y_u,fu,dx,m)
        M1=calc_M(u_last,v_last,eta,y_v,fv,dy,n)
        
        etaP=eta_last + K1*dt*0.5
        uP[:,1:m]=u_last[:,1:m] + L1*dt*0.5
        vP[1:n,:]=v_last[1:n,:] + M1*dt*0.5
        
        K2=calc_K(uP,vP,dx)
        L2=calc_L(uP,vP,etaP,y_u,fu,dx,m)
        M2=calc_M(uP,vP,etaP,y_v,fv,dy,n)
        
        etaP=etaP+ K2*dt*0.5
        uP[:,1:m]=uP[:,1:m] + L2*dt*0.5
        vP[1:n,:]=vP[1:n,:] + M2*dt*0.5
        
        K3=calc_K(uP,vP,dx)
        L3=calc_L(uP,vP,etaP,y_u,fu,dx,m)
        M3=calc_M(uP,vP,etaP,y_v,fv,dy,n)
        
        etaP=etaP+ K3*dt
        uP[:,1:m]=uP[:,1:m] + L3*dt
        vP[1:n,:]=vP[1:n,:] + M3*dt
        
        K4=calc_K(uP,vP,dx)
        L4=calc_L(uP,vP,etaP,y_u,fu,dx,m)
        M4=calc_M(uP,vP,etaP,y_v,fv,dy,n)
    
        eta=eta_last+ (1/6)*dt*(K1 + 2*K2 + 2*K3 + K4)
        u_u[:,1:m]=u_last[:,1:m] + (1/6)*dt*(L1 + 2*L2 + 2*L3 + L4)
        v_v[1:n,:]=v_last[1:n,:] + (1/6)*dt*(M1 + 2*M2 + 2*M3 + M4)
    
    
        energyTS[j]=Energy(u_u,v_v,eta,dx)
    
    
    return u_u, v_v, eta,dx,dt,nt,energyTS

    
    
    
    
    
    
    
    
    