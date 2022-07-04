from scipy.io import loadmat
from scipy.io import savemat

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import cmath
import pickle
from numba import jit,njit
from multiprocessing import Process, Manager
import multiprocessing
from scipy import optimize
import copy
import time
num_freq_pts=500 # generating nxn matrix for the Green functions
num_z_pts=1000# number of points in the z direction for simulation
pump_duration=0.1 #in ps
scale=1
scale_p=1
loss=0
wl= 793.6272586437883e-9#in m
#wl=820e-9
power=0.007 #in W
folder_name=str(power)+'W\\'
#folder_name=''
parts=0
approx=False
Rk4=True
def check_unique(v):
    for i in range (1,len(v)):
        assert(np.abs(v[i])<np.abs(v[i-1]))
def check_modes(omega_sig,u,v,w,S,n):
    out=u[:,0:n]*v[0:n]@w[0:n,:]
    plt.figure(10000)
    plt.pcolor(omega_sig,omega_sig,np.abs(out))
    plt.colorbar()
    plt.show()
    plt.figure(10001)
    plt.pcolor(omega_sig,omega_sig,np.abs(S-out))
    plt.colorbar()
    plt.show()

file_name = 'Green_Functions_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(pump_duration)+'ps'

if (num_z_pts%10==1):
    file_name+='_optimized'
if (loss>0):
    file_name+='_loss_'+str(loss)
if (power>0):
    file_name+='_'+str(power)+"W"
if (scale>0):
    file_name+='_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)
if (wl>0):
    file_name+='_'+str(wl)
if (approx):
    file_name+='_approx'
if (Rk4):
    file_name+='_substitution_rk4'
#declare variables
S=0
C=0
omega_sig=0
ssum=0
ddif=0
if (parts>0):
    S=[[0 for i in range(num_freq_pts)] for j in range(num_freq_pts)]
    C=[[0 for i in range(num_freq_pts)] for j in range(num_freq_pts)]
    for i in range(1,parts+1): #start indexing from 1
        data=pickle.load(open(file_name+'_part_'+str(i)+".pickle",'rb'))
        S+=data['S']
        C+=data['C']
        omega_sig=data['omega_sig']
else:
    data=pickle.load(open(folder_name+file_name+".pickle",'rb'))
    S=data['S']
    C=data['C']
    ssum=data['ssum']
    ddif=data['ddif']
    omega_sig=data['omega_sig']
# save as matlab file
# savemat("S_"+num_freq_pts+"x"+num_z_pts+".mat",{'S':S})

# plot S and C

# plt.figure(-1)
# plt.pcolor(omega_sig,omega_sig,np.abs(S))
# plt.colorbar()
# plt.show()
# plt.figure(0)
# plt.pcolor(omega_sig,omega_sig,np.abs(C))
# plt.colorbar()
# plt.show()
domg=omega_sig[1]-omega_sig[0]
us,vs,ws=np.linalg.svd(S)
if (num_z_pts>0):
    uc,vc,wc=np.linalg.svd(C)
    vc*=domg
    vs*=domg
#check_modes(omega_sig,us,vs,ws,S,1)
# check_unique(vs)
# check_unique(vc)
if (num_z_pts>0):
    plt.figure(101)
    plt.pcolor(omega_sig,omega_sig,np.abs(C))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    plt.colorbar()
    plt.show()
    plt.figure(102)
    plt.pcolor(omega_sig,omega_sig,np.abs(S))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    plt.colorbar()
    plt.show()
    plt.figure(103)
    plt.pcolor(omega_sig,omega_sig,np.abs(ssum))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    plt.colorbar()
    plt.show()
    plt.figure(104)
    plt.pcolor(omega_sig,omega_sig,np.abs(ddif))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    plt.colorbar()
    plt.show()
    plt.figure(105)
    plt.pcolor(omega_sig,omega_sig,np.abs(ddif+1j*ssum))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    plt.colorbar()
    plt.show()
    plt.figure(106)
    plt.pcolor(omega_sig,omega_sig,np.angle(S,deg=True))
    plt.colorbar()
    plt.title("S phase "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    plt.savefig(folder_name+"S phase, "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'_substitution_rk4.png', format='png')
    plt.show()
    plt.figure(107)
    plt.pcolor(omega_sig,omega_sig,np.abs(np.real(S)))
    plt.colorbar()
    plt.title("S real "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    #plt.savefig(folder_name+"S real, "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'_substitution_rk4.png', format='png')
    plt.show()
    plt.figure(108)
    plt.pcolor(omega_sig,omega_sig,np.imag(S))
    plt.colorbar()
    plt.title("S imag "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    #plt.savefig(folder_name+"S imag, "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'_substitution_rk4.png', format='png')
    plt.show()
plt.figure(0)
plt.plot(vs)
plt.show()
if (num_z_pts>0):
    plt.figure(1)
    plt.plot(20*np.log10(np.arcsinh(vs)))
    plt.show()
    # plt.figure(2)
    # plt.plot(np.arccosh(vc))
    # plt.show()
# plt.figure(3)
# plt.semilogy(np.abs(np.arccosh(vc)-np.arcsinh(vs))/np.arccosh(vc))
# plt.show()
# plt.figure(4)
# plt.semilogy(np.abs(np.arccosh(vc)-np.arcsinh(vs))/np.arcsinh(vs))
# plt.show()
# plt.figure(104)
# plt.semilogy(np.abs(np.arccosh(vc)-np.arcsinh(vs)))
# plt.show()
# for i in range (0,2):
#     plt.figure(200+i)
#     plt.plot(np.abs(us[:,i])**2)
#     plt.show()
#     plt.figure(100+i)
#     plt.plot(np.abs(uc[:,i])**2)
#     plt.show()

# plt.figure(5)
# plt.plot(np.abs(uc[:,0]))
# plt.show()
plt.figure(6)
plt.plot(omega_sig,np.abs(us[:,0]))
plt.show()
plt.figure(7)
plt.plot(omega_sig,np.abs(us[:,1]))
plt.show()
plt.figure(8)
plt.plot(omega_sig,np.real(us[:,2]))
plt.show()
plt.figure(9)
plt.plot(omega_sig,np.real(us[:,3]))
plt.show()
plt.figure(10)
plt.plot(omega_sig,np.real(us[:,4]))
plt.show()
plt.figure(11)
plt.plot(omega_sig,np.real(us[:,5]))
plt.show()

# plt.figure(13)
# plt.plot(np.real(ws[0,:]))
# plt.show()
# plt.figure(14)
# plt.plot(np.real(ws[1,:]))
# plt.show()
# ☻