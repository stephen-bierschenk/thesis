# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:45:08 2021

@author: Stephen
"""

import os
from datetime import datetime
from datetime import date
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog 
import math as m
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
from scipy.integrate import odeint 

global r,z,vgz,vgr,rho,eta,Ma_com,T_com,Dp,Vnp,Mnp,g,KdivM,Dnp,file,NPaxial,NPradial,z0,r0,Gas,length,tot_time,sub_dist,progress_var,counter,filename
counter = 0
filename = 'set'
window = tk.Tk()
window.geometry('400x400')
progress_var = tk.DoubleVar()
def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select a File", 
                                          filetypes = (("Text files", 
                                                        "*.txt*"), 
                                                       ("all files", 
                                                        "*.*")))
    # Change label contents 
    file_entry.configure(text="File Opened: "+os.path.basename(os.path.normpath(filename)))
def main():
    global counter,filename
    time1 = datetime.now()
    counter = 0
    print('Solver has started')
    progress_var.set(0)
    Dnp = float(Dnp_entry.get())
    rAg = float(rAg_entry.get())
    dense = float(Density_Var.get())
    NPaxial = float(NPaxial_entry.get())
    NPradial = float(NPradial_entry.get())
    r0 = float(r0_entry.get())
    z0 = float(z0_entry.get())
    Gas = Gas_Var.get()
    file = filename #'1_5mm10a5b_helium_500_1_torr.txt'
    length = int(length_entry.get())
    tot_time = float(tot_time_entry.get())
    method = Interp_Var.get()
    
    r0 = [-.005,-.004,-.003,-.002,-.001,0,.001,.002,.003,.004,.005]
    
    def Drag_1(Ma,Kn): # Calculates drag coefficient using data from Hogan Group
        lKn = m.log10(Kn)
        if lKn < 1:
            lKn = 1
          
        Ma = abs(Ma)
        S = Ma*m.sqrt(g/2)
    
        X = np.array([Ma,lKn])
        
        Mat = np.array([[-2.4007,1.7460,13.9004,-1.1289],
                        [-0.6223,2.1452,11.4413,-0.0787],
                        [-2.1060,1.4830,12.0261,1.3958],
                        [1.0772,-2.2110,-11.1729,0.1424],
                        [-0.0941,-0.2127,1.1092,-5.3498],
                        [-1.7974,-0.4568,6.2976,0.1227],
                        [-1.5316,1.1250,7.2065,0.3450],
                        [0.9180,-1.0428,-5.3537,0.6979],
                        [-2.6393,0.2688,5.4774,-0.8946],
                        [0.1141,1.0503,0.7186,-0.6304],
                        [-0.0921,2.2469,4.1520,-0.2392],
                        [-1.8866,0.3119,3.9608,2.5038],
                        [-0.8746,1.5139,4.6805,-0.3205],
                        [-1.6530,0.4458,3.5227,-1.4925],
                        [3.8852,3.6667,2.7319,0.0575],
                        [2.0425,0.4680,0.1396,2.6853],
                        [-1.0352,-0.8592,-6.4200,-1.5161],
                        [-2.9769,-0.4797,0.2721,1.6505],
                        [-1.1795,0.2254,0.3843,2.3026],
                        [-0.8688,0.3617,-0.0408,-2.7195]])
        W1 = Mat[:,0:2]
        W2 = Mat[:,3:]
        W2 = np.transpose([W2])
        W2 = W2[:,:,0]
        b1 = Mat[:,2]
        b2 = 2.9633
        
        Cd = W2.dot(np.tanh(W1.dot(X) + b1)) + b2
        Frac = (0.45*Kn + 0.38*(0.053*S + 0.639*m.sqrt(Kn*S)))/(Kn + 0.053*S + 0.639*m.sqrt(Kn*S))
        Denom = Ma/Kn * m.sqrt(g*m.pi/2) * (1 + 2*Kn*(1.257 + 0.4*m.exp(-0.55/Kn))+m.exp(-0.447*m.sqrt(Ma*Kn/m.sqrt(g)))*Frac)
        Cd = Cd*24/Denom
        return Cd
    
    def ode(t,y): # Solves ODEs for particle position and velocity
        global counter
        progress = t/tot_time
        progress_var.set(progress)
        window.update_idletasks()
        znp = y[3]
        rnp = y[1]
        
        if znp > sub_dist:
            if counter == 0:
                print('Particle hits substrate at ',t,' seconds')
                counter = 1
        # Calculate gas velocity and properties from COMSOL data
        points = np.column_stack((r,z))
        # method = 'linear'
        vgz_t = griddata(points, vgz, (rnp, znp), method=method)
        if np.isnan(vgz_t):
            vgz_t = griddata(points, vgz, (rnp, znp), method='nearest')
        vgr_t = griddata(points, vgr, (rnp, znp), method=method)
        if np.isnan(vgr_t):
            vgr_t = griddata(points, vgr, (rnp, znp), method='nearest')
        rho_t = griddata(points, rho, (rnp, znp), method=method)
        if np.isnan(rho_t):
            rho_t = griddata(points, rho, (rnp, znp), method='nearest')
        eta_t = griddata(points, eta, (rnp, znp), method=method)
        if np.isnan(eta_t):
            eta_t = griddata(points, eta, (rnp, znp), method='nearest')
        T_t = griddata(points, T_com, (rnp, znp), method=method)
        if np.isnan(T_t):
            T_t = griddata(points, T_com, (rnp, znp), method='nearest')
        
        # Calculate drag on particle
        ubar = m.sqrt(3/g)*m.sqrt(g*kdivM*T_t)
        lambda2 = 2*eta_t/(rho_t*ubar)
        vs = m.sqrt(g*kdivM*T_t)
        Kn = 2*lambda2/Dp
    
        Vg = np.array([vgz_t,vgr_t]) #gas velocity
        Vnp = np.array([y[2],y[0]]) # particle velocity
        Vdiff = np.subtract(Vg,Vnp)
        mag_Vdiff = m.sqrt(Vdiff[0:1]**2+Vdiff[1:]**2)
        Ma_r = (mag_Vdiff)/vs #relative mach number
        
        Cd = Drag_1(Ma_r,Kn) # Hogan drag coefficient
        
        a1 = -m.pi/8*rho_t*Dp**2*Cd/Mnp # Particle drag
        a2 = -m.pi/8*rho_t*Dp**2*Cd/Mnp
        
     # set of differential equations   
        dr_dt = a1*m.sqrt((y[0] - vgr_t)**2+(y[2]-vgz_t)**2)*(y[0] - vgr_t)
        dr = y[0]
        dz_dt = a2*m.sqrt((y[0] - vgr_t)**2+(y[2]-vgz_t)**2)*(y[2] - vgz_t)  # -vgz is correct
        dz = y[2]
        
        return [dr_dt,dr,dz_dt,dz]
    # Set the NP Diameter
    
    No = 6.022E+23  # Avogodro's no
    k = 1.38E-23  # Boltzman J/K
    
    # Set gas type

    
    if Gas == 'He':
        g = 1.6667
        Ma = 4  # He  gm/mole
        Ma = 0.001*Ma/No
        kdivM = k/Ma  # MKS
    elif Gas == 'Ar':
        g = 1.6667
        Ma = 39.95  # Ar  gm/mole
        Ma = 0.001*Ma/No
        kdivM = k/Ma  # MKS
    elif Gas == 'N2':
        g = 1.4
        Ma = 28.02  # N2  gm/mole
        Ma = 0.001*Ma/No
        kdivM = k*No/Ma  # cgs
    elif Gas == 'Air':
        g = 1.4
        Ma = 28.97
        Ma = 0.001*Ma/No
        kdivM = k/Ma 
    else:
        print('Invalid gas type')
        sys.exit(0)
    
    # import comsol gas data

    data = np.loadtxt(file)
    
    # num_rows, num_cols =data.shape
    # delete = []
    # for i in range(0,num_rows):
    #     if data[i,2] > 0:
    #         delete.append(i)
        
    # data = np.delete(data, delete, 0)
    # data = np.delete(data, 2, 1)
    
    
    # radial position, m
    r = data[:,1]
    # axial position, m
    z = data[:,0]
    sub_dist = max(z) # substrate total distance COMSOL
    # radial gas velocity, m/s
    vgr = data[:,2]
    # axial gas velocity, m/s
    vgz = data[:,3]
    # density, kg/m^3
    rho = data[:,4]
    # gas dynamic viscosity, Pa*s
    eta = data[:,5]
    # gas mach number
    Ma_com = data[:,6]
    # temperature, K
    T_com = data[:,7]
    
    # Nanoparticle properties calculated
    Dcm = Dnp*1E-7  # cm
    Dp = Dnp*1E-09  # meters
    # rAg = 5.01  # gm/cm^3 density of linbo3
    rho_aglom = dense*rAg/100
    rhoAg = rAg*1E3  # kg/m^2
    Vnp = (m.pi/6)*(Dcm)**3  # cm^3
    mnp = rho_aglom*Vnp  # 4.40E-17 gm
    Mnp = mnp/1000  # 4.40E-20 kg
    Vnp = Vnp*1E-6  # m^3
    
    # number of data points in ODE solver
    t_span = np.array([0,tot_time])
    times = np.linspace(t_span[0],t_span[1], length)
    
    # initial conditions for ODE solver
    y0 = np.array([NPradial,r0,NPaxial,z0])
    
    results = solve_ivp(ode, t_span, y0, method='RK45', t_eval = times)
    results_array = np.array([results.t,results.y[0],results.y[1],results.y[2],results.y[3]])
    
    i = len(results_array[4])
    check = np.arange(i)
    remove = np.empty(0, dtype=int)
    
    # for x in check:
    #     if results_array[4,x] > sub_dist:
    #         remove = np.append(remove, [x])
    # results_array = np.delete(results_array,remove,1)
    
    remove2 = np.empty(0, dtype=int)
    for x in check:
        if results_array[4,x] > (results_array[2,x]+0.01958386680220897)/2.747479187845152:
            remove2 = np.append(remove2, [x])
    results_array = np.delete(results_array,remove2,1)
    
    t = results_array[0]
    dr_dt = results_array[1]
    dr = results_array[2]
    dz_dt = results_array[3]
    dz = results_array[4]
    time2 = datetime.now()
    runtime = time2-time1
    print('Solver is finished.')
    print("Runtime:",runtime,"(hr:min:sec)")
    
    
    #fig, (ax1, ax2, ax3) = plt.subplots(3)
    #plt.subplots_adjust(hspace = .5)
    # ax1.plot(r,-z,'o',dr, -dz, 'r-')
    # ax1.set_title('Trajectory')
    # ax2.plot(dz, dz_dt, 'r-')
    # ax2.set_title('Axial Velocity')
    # ax3.plot(dz, dr_dt, 'r-')
    # ax3.set_title('Radial Velocity')
    
    #create outline of model
    coord = [[0,0],[0,.007],[-.003,.005],[-.003,.00035],[-.0034,.00035],[-.0034,.01],[-0.0100397,.01],[-0.0100397,.008],[-.0064,-.002],[-.0064,-.01],[-.0034,-.01],[-.0034,-.00035],[-.003,-.00035],[-.003,-.005],[0,-.007]]
    coord.append(coord[0]) #repeat the first point to create a 'closed loop'

    xs, ys = zip(*coord) #create lists of x and y values

    plt.figure(1)
    plt.title('Trajectory')
    plt.plot(dr, -dz, markersize=1) #r,-z,'o',
    plt.plot(ys,xs,color='b')

    plt.figure(2)
    plt.title('Axial Velocity')
    plt.plot(dz, dz_dt)

    plt.figure(3)
    plt.title('Radial Velocity')
    plt.plot(dz, dr_dt)
    plt.show()
    
    
def end():
    window.quit()
    window.destroy()

tk.Label(window, text="NP Diameter, nm").grid(row=1)
tk.Label(window, text="Material Density, g/cm^3").grid(row=2)
tk.Label(window, text="Aglomerate Density, %").grid(row=3)
tk.Label(window, text="File Explorer", fg = "blue").grid(row=4)
tk.Label(window, text="NP initial axial velocity, m/s").grid(row=5)
tk.Label(window, text="NP initial radial velocity, m/s").grid(row=6)
tk.Label(window, text="NP initial axial position, m").grid(row=7)
tk.Label(window, text="NP initial radial position, m").grid(row=8)
tk.Label(window, text="Gas type").grid(row=9)
tk.Label(window, text="# time steps for ODE solver output").grid(row=10)
tk.Label(window, text="Total time for ODE solver, s").grid(row=11)
tk.Label(window, text="Interpolation Method").grid(row=12)

Dnp_entry = tk.Entry(window)
Dnp_entry.insert(0,'200') # default nanoparticle default diameter

rAg_entry = tk.Entry(window)
rAg_entry.insert(0, '4.65') # default nanoparticle material density

Density_Var = tk.DoubleVar(window)
Density_entry = tk.Scale(window,variable = Density_Var,from_= 10, to=100, orient= tk.HORIZONTAL)
Density_Var.set(100) # default aglomerate density

file_entry = tk.Button(window,  
                        text = "Browse Files", 
                        command = browseFiles)
# file_entry = tk.Entry(window)
# file_entry.insert(0,'1_5mm10a5b_helium_500_1_torr.txt') # default file name

NPaxial_entry = tk.Entry(window)
NPaxial_entry.insert(0,'5') # default axial velocity

NPradial_entry = tk.Entry(window)
NPradial_entry.insert(0,'0') # default radial velocity

z0_entry = tk.Entry(window)
z0_entry.insert(0,'0') # default axial starting position

r0_entry = tk.Entry(window)
r0_entry.insert(0,'0.001') # default radial starting position


GasOptions = ["He","N2","Ar","Air"]
Gas_Var = tk.StringVar(window)
Gas_Var.set(GasOptions[3]) # default gas option
Gas_entry = tk.OptionMenu(window,Gas_Var,*GasOptions)

length_entry = tk.Entry(window)
length_entry.insert(0,'20000') # default number of ODE solver points

tot_time_entry = tk.Entry(window)
tot_time_entry.insert(0,'0.00065') # default time for ode solver

InterpolationOptions = ["nearest","linear","cubic"]
Interp_Var = tk.StringVar(window)
Interp_Var.set(InterpolationOptions[1]) # default interpolation method
Interp_entry = tk.OptionMenu(window,Interp_Var,*InterpolationOptions)

Dnp_entry.grid(row=1,column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
rAg_entry.grid(row=2,column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
Density_entry.grid(row=3,column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
file_entry.grid(row=4, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
NPaxial_entry.grid(row=5, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
NPradial_entry.grid(row=6, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
z0_entry.grid(row=7, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
r0_entry.grid(row=8, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
Gas_entry.grid(row=9, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
length_entry.grid(row=10, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
tot_time_entry.grid(row=11, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
Interp_entry.grid(row=12, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))

Dnp = float(Dnp_entry.get())
rAg = float(rAg_entry.get())
dense = float(Density_Var.get())
NPaxial = float(NPaxial_entry.get())
NPradial = float(NPradial_entry.get())
r0 = float(r0_entry.get())
z0 = float(z0_entry.get())
Gas = Gas_Var.get()
file = filename #'1_5mm10a5b_helium_500_1_torr.txt'
length = int(length_entry.get())
tot_time = float(tot_time_entry.get())
method = Interp_Var.get()

progressbar = ttk.Progressbar(window, variable=progress_var, maximum=1)
progressbar.grid(row=13,column=0)


window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=2)
for x in range(0,20):
    window.rowconfigure(x, weight=1)

tk.Button(window, text="Run", command=main).grid(row=14,column=0)
tk.Button(window, text="Quit", command=end).grid(row=14,column=1)
window.mainloop()
print('this should print after mainloop is ended')