from scipy.io import loadmat
from scipy.io import savemat
import scipy.linalg as la
from scipy.interpolate import interp1d
import scipy
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
num_z_pts=50000# number of points in the z direction for simulation
pump_duration=0.1 #in ps
scale=1
scale_p=1
loss=0
wl= 793.6272586437883e-9#in m
#wl=820e-9
power=0.005 #in W
folder_name=str(power)+'W\\'
#folder_name=''
parts=0
approx=False
Rk4=False
def check_unique(v):
    for i in range (1,len(v)):
        if (not(np.abs(v[i])<np.abs(v[i-1]))):
            print(i)
            return
    print('all good')
def check_modes(omega_sig,u,v,w,S,n):
    out=u[:,0:n]*v[0:n]@w[0:n,:]
    plt.figure(20000)
    plt.pcolor(omega_sig,omega_sig,np.abs(out))
    plt.colorbar()
    plt.show()
    plt.figure(20001)
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
C=C.T
S=S.T
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
ws=ws.T.conj()
if (num_z_pts>0):
    uc,vc,wc=np.linalg.svd(C)
    vc*=domg
    vs*=domg
    wc=wc.T.conj()
# plt.figure(101)
# plt.pcolor(omega_sig,omega_sig,np.abs(C@(np.conj(C).T)-S@(np.conj(S).T)))
# #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
# plt.colorbar()
# plt.show()
# plt.figure(102)
# plt.pcolor(omega_sig,omega_sig,np.abs((C@S).T-C@S))
# #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
# plt.colorbar()
# plt.show()
# plt.figure(103)
# plt.pcolor(omega_sig,omega_sig,np.abs(C@(np.conj(C).T)))
# #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
# plt.colorbar()
# plt.show()
test=np.array([[1,2,3],[4,5,6],[7,8,9]])
x=[1,2,3,4]
# plt.figure(-1)
# plt.pcolor(x,x,test)
# #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
# plt.colorbar()
# plt.show()
if (num_z_pts>1e10):

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

    plt.figure(106)
    plt.pcolor(omega_sig,omega_sig,np.angle(S,deg=True))
    plt.colorbar()
    plt.title("S phase "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
    #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    plt.savefig(folder_name+"S phase, "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'_substitution_rk4.png', format='png')
    plt.show()
    # plt.figure(107)
    # plt.pcolor(omega_sig,omega_sig,np.abs(np.real(S)))
    # plt.colorbar()
    # plt.title("S real "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
    # #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    # #plt.savefig(folder_name+"S real, "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'_substitution_rk4.png', format='png')
    # plt.show()
    # plt.figure(108)
    # plt.pcolor(omega_sig,omega_sig,np.imag(S))
    # plt.colorbar()
    # plt.title("S imag "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
    # #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
    # #plt.savefig(folder_name+"S imag, "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'_substitution_rk4.png', format='png')
    # plt.show()
# plt.figure(0)
# plt.plot(vs)
# plt.show()
if (num_z_pts>0):
    plt.figure(1)
    plt.title('mode distribution')
    plt.plot(20*np.log10(np.arcsinh(vs)))
    plt.ylabel('dB')
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

##joint decomposition transformation
u=np.zeros(S.shape,dtype=np.complex128)
u_diag=np.zeros(len(omega_sig),dtype=np.complex128)
for i in range(len(omega_sig)):
    u_diag[i]=np.angle(us[0,i]/uc[0,i])
np.fill_diagonal(u,np.exp(1j*u_diag))
uc=uc@u
wc=wc@u
uc_old=uc
## Takagi
G=(np.conj(wc).T)@np.conj(ws)
D1=scipy.linalg.sqrtm(G)

uc=uc@D1
wc=wc@D1
us=us@D1
ws=ws@D1


# plt.figure(1001)
# plt.pcolor(np.log10(np.abs(S-us@np.diag(vs/domg)@ws.conj().T)))
# plt.colorbar()
# plt.show()
# plt.figure(1002)
# plt.pcolor(np.log10(np.abs(C-uc@np.diag(vc/domg)@wc.conj().T)))
# plt.colorbar()
# plt.show()
#
# plt.figure(1003)
# plt.pcolor(np.log10(np.abs(S)))
# plt.colorbar()
# plt.show()
# plt.figure(1004)
# plt.pcolor(np.log10(np.abs(C)))
# plt.colorbar()
# plt.show()
# plt.figure(1005)
# plt.pcolor(np.log10(np.abs(us@np.diag(vs/domg)@ws.conj().T)))
# plt.colorbar()
# plt.show()
# plt.figure(1006)
# plt.pcolor(np.log10(np.abs(uc@np.diag(vc/domg)@wc.conj().T)))
# plt.colorbar()
# plt.show()
# plt.figure(1007)
# plt.pcolor(np.abs(us@np.diag(vs/domg)@ws.conj().T))
# plt.colorbar()
# plt.show()
# plt.figure(1008)
# plt.pcolor(np.abs(uc@np.diag(vc/domg)@wc.conj().T))
# plt.colorbar()
# plt.show()
# plt.figure(1009)
# plt.pcolor(np.log10(np.abs(G-D1@D1)))
# plt.colorbar()
# plt.show()
# plt.figure(1010)
# plt.pcolor(np.log10(np.abs(D1-D1.T)))
# plt.colorbar()
# plt.show()
## modes from U:
plt.figure(6)
plt.title('first mode')
plt.plot(omega_sig,(np.abs(us[:,0]-uc[:,0])))
plt.plot(omega_sig,(np.abs(us[:,0])))

plt.show()
plt.figure(7)
plt.title('second mode')
plt.plot(omega_sig,np.square(np.abs(us[:,1]-uc[:,1])))
plt.plot(omega_sig,np.square(np.abs(us[:,1])))
plt.show()
plt.figure(8)
plt.title('third mode')
plt.plot(omega_sig,np.square(np.abs(us[:,2]-uc[:,2])))
plt.plot(omega_sig,np.square(np.abs(us[:,2])))
plt.show()
plt.figure(9)
plt.plot(omega_sig,np.square(np.abs(us[:,3]-uc[:,3])))
plt.plot(omega_sig,np.square(np.abs(us[:,3])))
plt.show()
plt.figure(10)
plt.plot(omega_sig,np.square(np.abs(us[:,4]-uc[:,4])))
plt.plot(omega_sig,np.square(np.abs(us[:,4])))
plt.show()
plt.figure(11)
plt.plot(omega_sig,np.square(np.abs(us[:,5]-uc[:,5])))
plt.plot(omega_sig,np.square(np.abs(us[:,5])))
plt.show()

#check that transformation is okay
plt.figure(206)
plt.title('first mode')
plt.plot(omega_sig,(np.abs(us[:,0])-np.abs(uc_old[:,0])))

plt.show()
plt.figure(207)
plt.title('second mode')
plt.plot(omega_sig,np.square(np.abs(us[:,1])-np.abs(uc_old[:,1])))
plt.show()
plt.figure(208)
plt.title('third mode')
plt.plot(omega_sig,np.square(np.abs(us[:,2])-np.abs(uc_old[:,2])))
plt.show()

## modes from W:
plt.figure(16)
plt.title('first mode difference')
plt.plot(omega_sig,(np.abs(ws[:,0]-np.conj(wc[:,0]))))
#plt.plot(omega_sig,(np.abs(us[:,0])))
plt.plot(omega_sig,(np.abs(ws[:,0])))

plt.show()
plt.figure(17)
plt.title('second mode')
plt.plot(omega_sig,np.square(np.abs(ws[:,1]-np.conj(wc[:,1]))))
plt.plot(omega_sig,np.square(np.abs(ws[:,1])))
plt.show()
plt.figure(18)
plt.title('third mode')
plt.plot(omega_sig,np.square(np.abs(ws[:,2]-np.conj(wc[:,2]))))
plt.plot(omega_sig,np.square(np.abs(ws[:,2])))
plt.show()
plt.figure(19)
plt.plot(omega_sig,np.square(np.abs(ws[:,3]-np.conj(wc[:,3]))))
plt.plot(omega_sig,np.square(np.abs(ws[:,3])))
plt.show()
plt.figure(20)
plt.plot(omega_sig,np.square(np.abs(ws[:,4]-np.conj(wc[:,4]))))
plt.plot(omega_sig,np.square(np.abs(ws[:,4])))
plt.show()
plt.figure(21)
plt.plot(omega_sig,np.square(np.abs(ws[:,5]-np.conj(wc[:,5]))))
plt.plot(omega_sig,np.square(np.abs(ws[:,5])))
plt.show()
## mode magnitude
plt.figure()
plt.title('mode difference')
plt.plot((np.arccosh(vc[0:300])-np.arcsinh(vs[0:300]))/np.abs(vc[0:300]))
plt.show()

plt.figure()
plt.title('mode arccosh')
plt.plot(np.arccosh(vc[0:300]))
plt.show()

plt.figure()
plt.title('mode arcsinh')
plt.plot(np.arcsinh(vs[0:300]))
plt.show()
plt.figure()
plt.title('mode cosh')
plt.plot(vc)
plt.show()
plt.figure()
plt.title('mode sinh')
plt.plot(vs)
plt.show()
# ☻