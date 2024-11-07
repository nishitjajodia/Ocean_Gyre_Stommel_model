# -*- coding: utf-8 -*-
"""
Functions
"""
import numpy as np
import matplotlib.pyplot as plt
from Constants import*
from funcs import*

def f1(x):
    '''
    Calculates f1(x)
    Parameters
    ----------
    x : float

    Returns
    -------
    f1 : float
        f1(x).
    '''
    f1=np.pi*( 1 + ( ((np.exp(a)-1)*np.exp(b*x) + (1-np.exp(b))*np.exp(a*x))/(np.exp(b)-np.exp(a))) )
    return f1

def f2(x):
    '''
    Calculates f2(x)
    Parameters
    ----------
    x : float

    Returns
    -------
    f2 : float
        f2(x).
    '''
    f2=( ((np.exp(a)-1)*b*np.exp(b*x) + (1-np.exp(b))*a*np.exp(a*x))/(np.exp(b)-np.exp(a)))
    return f2

def tau(y,L):
    tau_x= -0.2*np.cos(np.pi*y/L)
    tau_y=0
    return tau_x, tau_y

def dudx(dx,u):
    dudx=(u[:,1:]-u[:,:-1])/dx
    return dudx

def dvdy(dy,v):
    dvdy=(v[:-1,:]-v[1:,:])/dy
    return dvdy

def detadx(dx,eta):
    dedx= (eta[:,1:]-eta[:,:-1])/dx
    return dedx

def detady(dy,eta):
    dedy= (eta[:-1,:]-eta[1:,:])/dy
    return dedy

def calc_Vint(v_v):
    v_int=(v_v[:-1,:-1]+v_v[:-1,1:]+v_v[1:,:-1]+v_v[1:,1:])/4
    
    return v_int

def calc_Uint(u_u):
    u_int=(u_u[:-1,:-1]+u_u[:-1,1:]+u_u[1:,:-1]+u_u[1:,1:])/4
    
    return u_int

def calc_uONeta(u_u):
    uONeta=(u_u[:,:-1]+u_u[:,1:])/2
    return uONeta

def calc_vONeta(v_v):
    vONeta=(v_v[:-1,:]+v_v[1:,:])/2
    return vONeta

def calc_K(u_last,v_last,dx):
    K= -H*(dudx(dx,u_last)+dvdy(dx,v_last))  
    return K

def calc_L(u_last,v_last,eta,y_u,fu,dx,m):
    v_int=calc_Vint(v_last)
    L1=fu*v_int - g*detadx(dx,eta) - gamma*u_last[:,1:m] + tau(y_u[:,1:m],L)[0]/(rho*H)
    return L1

def calc_M(u_last,v_last,eta,y_v,fv,dy,n):
    u_int=calc_Uint(u_last)
    M=- fv*u_int - g*detady(dy,eta) - gamma*v_last[1:n,:] + tau(y_v[1:n,:],L)[1]/(rho*H)
    return M

def AnalyticalonArakawa(u_st,v_st,eta_st):
    ust1=(u_st[:-1,:]+u_st[1:,:])/2
    vst1=(v_st[:,:-1]+v_st[:,1:])/2
    etast1=(eta_st[:-1,:-1]+eta_st[:-1,1:]+eta_st[1:,:-1]+eta_st[1:,1:])/4
    
    return ust1,vst1,etast1
#%% plots

def contplot(u,v,eta,title,isdiff=False):
    '''
    Generates 2d contour plots
    Parameters
    ----------
    u : 2d array
    
    v : 2d array
        .
    eta : 2d array
        
    title : char string
        
    isdiff : Boolean, optional
        Is it a difference plot?. The default is False.

    Returns
    -------
    None.

    '''
    data=[np.flip(u,axis=0),v,eta]
    fig, ax = plt.subplots(1,3,sharey=True,figsize=(15,5))
    fig.suptitle(title)
    extent=[0,L,0,L]
    if isdiff==True:
        labels=['$u^1 (m/s)$','$v^1 (m/s)$','$\eta^1 (m)$']
    else:
        labels=['$u (m/s)$','$v (m/s)$','$\eta (m)$']
    ax[0].set_ylabel('Y (m)')
    for i in range(3):  
        ax[i].set_title(labels[i])
        ax[i].set_xlabel('X (m)')
        plot=ax[i].contourf(data[i],levels=25,extent=extent)
        plt.colorbar(plot,ax=ax[i],location='bottom')

    plt.tight_layout()
    plt.show()
    
def ploty(q,dx,position,Ylabel):
    size=q.shape[0]
    if size%2==0:
        y=np.linspace(L-dx/2,dx/2,size)
    else:
        y=np.linspace(L,0,size)
    
    if position==0:  # western boundary
        plt.plot(y,q[:,0],label='western boundary')
    if position==1:
        plt.plot(y,q[:,-1],label='eastern boundary')  # eastern boundary
    plt.xlabel('Y (m)')
    plt.ylabel(Ylabel)
    plt.grid()
    
def plotx(q,dx,position,Ylabel):
    size=q.shape[1]   
    if size%2==0:
        x=np.linspace(dx/2,L-dx/2,size)
    else:
        x=np.linspace(0,L,size)
    if position==0:  # northern boundary
        plt.plot(x,q[0,:],label='northern boundary')
    if position==1: # middle
        if size%2==0:
            plt.plot( x,(q[int(size/2),:]+q[int(size/2 -1),:])/2 )
        if size%2==1:
            plt.plot(x,q[int((size-1)/2),:])
    if position==2:   # southern boundary
        plt.plot(x,q[-1,:],label='southern boundary')  
    plt.xlabel('X (m)')
    plt.ylabel(Ylabel)
    plt.grid()
#%% Energy

def Energy(u_u,v_v,eta,dx):
    '''
    Calculates total energy of system

    Parameters
    ----------
    u_u :  2d array
    
    v_v :  2d array
    
    eta :  2d array
    
    dx : float
        spatial resolution in m.

    Returns
    -------
    energy :float
        total energy of system in Joules.

    '''
    u=calc_uONeta(u_u)
    v=calc_vONeta(v_v)
    energy=np.sum(0.5*rho*(H*(u**2 + v**2) +(g*eta**2))*(dx**2))
    return energy

def steadyfinder(energy,dt):
    '''
    Finds steady state for given energy time series

    Parameters
    ----------
    energy : 1d array
       energy time series.
    dt : float
        time step in seconds.

    Returns
    -------
    steadytime : float
        steady state time in days.

    '''
    for i in range(1,len(energy)):
        diff=energy[i]-energy[i-1]
        
        if diff<=1:
            steadytime=(i*dt)/(3600*24)
            print("Steady state is reached in ",steadytime,"days")
            break
    return steadytime

