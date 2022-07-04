from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import cmath
import pickle
from numba import jit,njit
from multiprocessing import Process, Manager
import multiprocessing
from scipy import optimize
import copy
import time
import os
import sys
os.environ["QT_API"] = "pyqt5"
# Bloch-Messiah decomposition, work with angular frequency (rad/s)
# Read in effective index value
# No sub index loop test
c=3e8
nm = 1e9
wl=793.6272586437883e-9 #original wavelength
# wl=775e-9
#wl=800e-9
omega_p0=2*np.pi*c/wl
deg1=5
deg2=5
# 700nm-900nm for Bragg modes
# 700nm- for TIR modes
def parse_data(plot_flag):
    data11= loadmat('775nm,TM4,below.mat') #1 is pump
    data21 = loadmat('1550nm,TE4,below(2).mat') # 2 is signal
    data12 = loadmat('775nm,TM4,above.mat') #1 is pump
    data22 = loadmat('1550nm,TE4,above.mat') # 2 is signal

    omg11 = 2*np.pi*data11['f']
    omg12 = 2*np.pi*data12['f']
    omg21 = 2*np.pi*data21['f']
    omg22 = 2*np.pi*data22['f']

    n1data1 = np.abs(data11['neff'])
    n1data2 = np.abs(data12['neff'])

    n2data1 = np.abs(data21['neff'])
    n2data2 = np.abs(data22['neff'])

    omg1 = np.concatenate((omg11[:,0],omg12[:,0]))
    omg1 = np.sort(np.unique(omg1))
    oomg1=omg1[40:51]
    omg2 = np.concatenate((omg21[:, 0], omg22[:, 0]))
    omg2 = np.sort(np.unique(omg2))
    oomg2=omg2[34:62]
    n1data = np.concatenate((n1data1[:,0],n1data2[:,0]))
    n1data = np.sort(np.unique(n1data))
    nn1data=n1data[40:51]
    n2data = np.concatenate((n2data1[:, 0], n2data2[:, 0]))
    n2data = np.sort(np.unique(n2data))
    nn2data=n2data[34:62]
    lambda1 = 2*np.pi*c/omg1*nm
    lambda2 = 2*np.pi*c/omg2*nm

    # n1 = interp1d(omg1, n1data,kind='cubic')
    # n2 = interp1d(omg2, n2data)

    n1 = np.polyfit(oomg1-omega_p0, nn1data, deg=deg1)#pump
    n2 = np.polyfit(oomg2-omega_p0/2, nn2data, deg=deg2)#signal
    if plot_flag == True:
        plt.figure(-1)
        plt.plot(omg1, n1data, 'o', omg1, poly_val(n1,omg1,1), '-')
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Effective index')
        plt.title('TM')
        plt.legend(['data', 'interpolation'], loc='best')
        plt.show()
        plt.figure(-2)
        plt.plot(omg2, n2data, 'o', omg2, poly_val(n2,omg2,2), '-')
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Effective index')
        plt.title('TE')
        plt.legend(['data', 'interpolation'], loc='best')
        #plt.figure(figsize=(3, 3))
        plt.show()
    return omg1,omg2, n1,n2

def poly_val(coef, omg, choice):
    ans=0.0
    omg0=omega_p0
    if (choice==2):
        omg0/=2
    for i,C in enumerate(reversed(coef)):
        ans+=C*(omg-omg0)**i
    return ans

omg1,omg2,n1,n2 = parse_data(False)
def kp(omg,fo=n1[-2],zo=n1[-1]):
    tmp=[n1[-2],n1[-1]]
    n1[-2]=fo
    n1[-1]=zo
    ans=poly_val(n1,omg,1)*omg/c
    n1[-2]=tmp[0]
    n1[-1]=tmp[1]
    return ans
def k(omg,fo=n2[-2],zo=n2[-1]):
    tmp=[n2[-2],n2[-1]]
    n2[-2]=fo
    n2[-1]=zo
    ans=poly_val(n2,omg,2)*omg/c
    n2[-2]=tmp[0]
    n2[-1]=tmp[1]
    return ans
def deltak(omgi,omgs,fo1=n1[-2],fo2=n2[-2],zo1=n1[-1],zo2=n2[-1]):
    return kp(omgi+omgs,fo1,zo1)-k(omgi,fo2,zo2)-k(omgs,fo2,zo2)
def Ep(E0,omega,omgp,pump_duration): #omgp is the central frequency, dt is the pump's FWHM
    pump_bw=1.177*np.sqrt(2)/pump_duration
    pump_field = E0*1/(np.sqrt(2*np.pi)*pump_bw)*np.exp(-((omega-omgp)**2/(2*pump_bw**2)))
    return pump_field
def phasematching(omgi, omgs, leff,fo1=n1[-2],fo2=n2[-2],zo1=n1[-1],zo2=n2[-1]):
    return np.sinc(deltak(omgi,omgs,fo1,fo2,zo1,zo2)*leff/2/np.pi)
def f(E0,omgi,omgs,omgp,pump_duration,leff,fo1=n1[-2],fo2=n2[-2],zo1=n1[-1],zo2=n2[-1]):
    return Ep(E0,omgs+omgi,omgp,pump_duration)*np.sinc(deltak(omgi,omgs,fo1,fo2,zo1,zo2)*leff/2/np.pi)

@jit(nopython=True)
def z_propagation(omega_sig,a,lp,lp_length,k_sig,k_pump,E_pump,dz,Leff):
    # Step Through Each Length Index
    cur=0
    nextt=0
    for z_idx in range(0,lp_length-1):
        nextt=(cur^1)
        a[cur, :] = a[cur, :] * np.exp(1j * k_sig * dz/2)
        for omega1_idx, omega in enumerate(omega_sig):
            #a_increment = field_increment(omega_sig, omega1_idx, a, z_idx,lp[z_idx],E_pump)
            z_val = lp[z_idx]
            #--------------------------------------------
            integrand = np.exp(1j * k_pump[:, omega1_idx] * (z_val - Leff / 2)) * E_pump[:, omega1_idx] * np.conjugate(a[cur,:])
            domg = omega_sig[1] - omega_sig[0]
            nonlin_term = np.sum(integrand) * domg
            a_increment = 1 / (Lnl * E0) * nonlin_term
            #--------------------------------------------
            a[nextt, omega1_idx] = a[cur, omega1_idx] + a_increment * dz
        a[nextt, :] = a[nextt, :] * np.exp(1j * k_sig * dz/2)
        cur=nextt
    return a[cur,:]
# @jit(nopython=True) # can't jit
def get_greens_functions_multicore(omega_sig,k_sig,k_pump,E_pump,summ,diff,dz,omgn1,omgn2,scale_p,scale):

    k_pump =  kp(omega_pump,fo=n1[-2]*scale_p)
    k_sig = k(omega_sig, fo=n2[-2]*scale)
    Lnl = (8*c**2*k(omega_p/2,fo=n2[-2]*scale))/(omega_p**2*deff*E0) # in 1/m
    Leff=leff(scale_p,scale)
    print(Leff)
    dz=Leff/(num_z_pts)

    lp=np.arange(0,Leff,dz) #L partitioned
    # a = np.zeros((len(lp), len(omega_sig)), dtype=complex)
    a = np.zeros((2, len(omega_sig)), dtype=complex)
    for omega_index in range(omgn1,omgn2):
        print(str(omega_index)+'/'+str(omgn2))
        #f.write(str(omega_index)+'/'+str(omgn2))
        sys.stdout.flush()
        a[0, omega_index] = 1 / domg
        #a_out = z_propagation(omega_sig, a, lp, len(lp), k_sig, k_pump, E_pump,dz)
        #mag=cmath.polar(a_out[-1, omega_index])[0]
        #a_out[-1, omega_index] =0
        #a_out[-1, omega_index] *=(mag-1 / domg)/mag
        summ[omega_index] = z_propagation(omega_sig, a, lp, len(lp), k_sig, k_pump, E_pump,dz,Leff)#a_out[-1, :]
        a[0,:]=0
        a[0, omega_index] = 1j / domg
        #a_out =
        #mag=cmath.polar(a_out[-1, omega_index])[0]
        #a_out[-1, omega_index] =0
        #a_out[-1, omega_index] *=(mag-1 / domg)/mag
        diff[omega_index] = z_propagation(omega_sig, a, lp, len(lp), k_sig, k_pump, E_pump,dz,Leff)#a_out[-1, :]
        a[0,:]=0

# @jit(nopython=True)
# def get_greens_functions(omega_sig,a,k_sig,k_pump,E_pump,summ,diff,dz):#not used
#     for omega_index, omega in enumerate(omega_sig):
#         a[0, omega_index] = 1 / domg
#         a_out = z_propagation(omega_sig, a, lp, len(lp), k_sig, k_pump, E_pump,dz)
#         a_out[-1, omega_index] = 0
#         summ[omega_index, :] = a_out[-1, :]
#         a[0, omega_index] = 1j / domg
#         a_out = z_propagation(omega_sig, a, lp, len(lp), k_sig, k_pump, E_pump,dz)
#         a_out[-1, omega_index] = 0
#         diff[omega_index, :] = a_out[-1, :]
#         a[0, omega_index] = 0
#     S = (summ + diff) / (1 + 1j)
#     G = (summ - diff) / (1 + 1j)
#     return S,G

## Define Frequency
start_time = time.time()
wl=793.6272586437883e-9 #central wavelength for this particular simulation
# wl=775e-9
#wl=800e-9
omega_p=2*np.pi*c/wl
#lambda_1 = 2*np.pi*c/omg1[:]
#lambda_2 = 2*np.pi*c/omg2[:]
#4350s for freq-500, z-10000
num_freq_pts=500 # generating n+1 x n+1 matrix for the Green functions
num_z_pts=4000 # number of points in the z direction for simulation

L=0.001 # device effective length in m




pump_duration=5e-12
pump_bw = 1.177 * np.sqrt(2) / pump_duration
#omega_p=2*np.pi*c/791.5e-9
#omega_p=2*np.pi*c/793.6272586437883e-9
epsilon0 = 8.85e-12
freq_scale = 2500

deff = 370e-12 # in meter/volt

## effective length, define variables related to z
power_avg_in=0.005 #in W
power_avg_in=0.25 #in W
area=4.9*1e-12 #m^2
deltat=1/(80e6) #period
power_peak=power_avg_in*deltat/(1.06484*pump_duration)
E0=np.sqrt(8*np.pi**2*power_peak/(c*epsilon0*area)) #np.sqrt(2*np.pi)*pump_bw*np.sqrt(power_peak/(epsilon0*c*area))
I_p0=1/2*epsilon0*c*E0**2/(4*np.pi**2)
alpha_wave3_o = 500 #linaer loss (m^-1)
a2_wave3_o =1.15e-12 #TPA (m/W)

scale=100
scale_p=100
vg_p=1/(n1[-2]*omega_p/c+n1[-1]/c)
vg=1/(n2[-2]*omega_p/2/c+n2[-1]/c)

def I_p(z):
    return I_p0*alpha_wave3_o*np.exp(-alpha_wave3_o*z)/(alpha_wave3_o+a2_wave3_o*I_p0*(1-np.exp(-alpha_wave3_o*z)))
def func(Leff):
    return Leff-(1-np.exp(-(alpha_wave3_o+a2_wave3_o*I_p(L)/1)*L))/(alpha_wave3_o+a2_wave3_o*I_p(Leff)/1)
def leff(scale_p, scale): #incorporates the walk off length
    #global I_p0
    #I_p0=power_in
    sol = optimize.root(func,L, method='hybr')
    n1cpy=np.array(n1)
    n2cpy=np.array(n2)
    n1cpy[-2]*=scale_p
    n2cpy[-2]*=scale
    n1d=np.polyder(n1cpy)
    n2d=np.polyder(n2cpy)

    vvg_p=1/(poly_val(n1d,omega_p,1)*omega_p/c+poly_val(n1cpy,omega_p,1)/c)
    vvg=1/(n2[-2]*scale*omega_p/2/c+n2[-1]/c)
    vvg=1/(poly_val(n2d,omega_p/2,2)*omega_p/2/c+poly_val(n2cpy,omega_p/2,2)/c)
    Lwo=pump_duration/np.abs(1/vvg_p-1/vvg)
    return min(Lwo,sol.x[0])
Leff=0
dz=0

lp=0 #np.arange(0,Leff,dz) #np.arange(0,Leff,dz) #L partitioned
##saving data

omega_signal_c = omega_p/2
omega_sig = np.linspace(omega_signal_c - pump_bw*freq_scale,omega_signal_c+pump_bw*freq_scale,num_freq_pts)
omega_pump = np.array(np.meshgrid(omega_sig,omega_sig))[0] + np.array(np.meshgrid(omega_sig,omega_sig))[1]
k_sig = 0 #k(omega_sig, fo=n2[-2]*scale)
k_pump =0 #  kp(omega_pump,fo=n1[-2]*scale_p)
E_pump = Ep(E0,omega_pump,omega_p,pump_duration)
Lnl = (8*c**2*k(omega_p/2))/(omega_p**2*deff*E0) # in 1/m
domg = omega_sig[1] - omega_sig[0]

#summ = np.zeros((len(omega_sig),len(omega_sig)), dtype=complex)
#diff = np.zeros((len(omega_sig),len(omega_sig)), dtype=complex)
init_list=[0]*(len(omega_sig))
# part=13
# omg_start=0
# omg_start=omg_start+48
# omg_start=omg_start+48
# omg_start=omg_start+8
# omg_start=omg_start+48
# omg_start=omg_start+8
# omg_start=omg_start+24
# omg_start=omg_start+48
# omg_start=omg_start+48
# omg_start=omg_start+48
# omg_start=omg_start+48
# omg_start=omg_start+8
# omg_start=omg_start+24
# omg_end=omg_start+48
part=0
omg_start=0
omg_end=len(omega_sig)
figure_num=0
## start solving
#should convert variables to manager.list/value
arr=np.round(np.arange(0.7,1.31,0.1),2)
#arr=np.array([1])
arr_p=np.array(arr)
# arr=[1.1]
# arr_p=[0.9]
arr_p=np.array([1])
arr=np.array([1])
gains=np.zeros((len(arr),len(arr)))
schmidt_n=np.zeros((len(arr),len(arr)))
#f = open('0.005W'+"\\"+"progress.txt", "w")
if __name__=='__main__':
    for i,val_i in enumerate(arr_p):
        for j,val_j in enumerate(arr):
            scale_p=val_i
            scale=val_j

            processes = multiprocessing.cpu_count()*2//4 #number of cores in the CPU
            #processes=1
            manager=Manager()
            summ=manager.list(init_list)
            diff=manager.list(init_list)
            for l in range(len(omega_sig)):
                summ[l]=[0]*(len(omega_sig))
                diff[l]=[0]*(len(omega_sig))
            #summ=[[0]*(len(omega_sig)) for j in range(len(omega_sig))] #doesn't work
            #diff=[[0]*(len(omega_sig)) for j in range(len(omega_sig))]
            index=np.linspace(omg_start,omg_end,processes+1,dtype=int, endpoint=True)
            p=[]

            for l in range(processes):
                p.append(Process(target=get_greens_functions_multicore, args=(omega_sig,k_sig,k_pump,E_pump,summ,diff,dz,index[l],index[l+1],val_i,val_j,)))
                p[l].start()

            for l in range(processes):
                p[l].join()
            ssum=np.asarray(summ, dtype=np.complex128)
            ddif=np.asarray(diff, dtype=np.complex128)
            C=1/2*(ssum-1j*ddif)
            S=1/2*(ssum+1j*ddif)

            u,v,w=np.linalg.svd(S)
            v*=domg
            first_mode=np.arcsinh(v[0])
            gains[i][j]=first_mode
            normalize=np.sqrt(np.sum(np.arcsinh(v)**2))
            vs_new=np.arcsinh(v)/normalize
            schmidt_n[i][j]=np.sum(vs_new**4)
            to_save = {'omega_sig':omega_sig ,'C':C, 'S':S, 'ssum': ssum, 'ddif': ddif, 'omgn1':omg_start, 'omgn2':omg_end, 'scale_p':scale_p, 'scale':scale, 'fm':first_mode, 'ttime':(time.time() - start_time)}
            folder_name=str(power_avg_in)+'W\\'
            os.makedirs(folder_name, exist_ok=True)
            file_name = 'Green_Functions_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(pump_duration*1e12)+"ps_"+str(power_avg_in)+"W"+'_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)
            if part>0:
                file_name+='_part_'+str(part)
            with open(folder_name+file_name+'.pickle','wb') as handle:
                pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

            #plt.pcolor(2*np.pi*c/omega_sig*1e9,2*np.pi*c/omega_sig*1e9,np.abs(S))
            plt.figure(figure_num)
            plt.pcolor(omega_sig,omega_sig,np.abs(S))
            plt.colorbar()

            plt.xlabel(r"$\omega$ (rad/s)")
            plt.ylabel(r"$\omega'$ (rad/s)")
            plt.title(r"$S(\omega,\omega')$ "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))

            plt.savefig(folder_name+"S "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power_avg_in)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'.png', format='png')
            #plt.show()

            plt.figure(figure_num+1)
            plt.pcolor(omega_sig,omega_sig,np.abs(C))
            plt.colorbar()

            plt.xlabel(r'$\omega$ (rad/s)')
            plt.ylabel(r"$\omega'$ (rad/s)")
            plt.title(r"$C(\omega,\omega')$ "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
            plt.savefig(folder_name+"C "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power_avg_in)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'.png', format='png')
            #plt.show()

            plt.figure(figure_num+2)
            plt.title("phasematching function "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
            plt.pcolor(omega_sig,omega_sig,phasematching(omega_sig[:,None],omega_sig[None,:],leff(val_i,val_j),fo1=n1[-2]*val_i, fo2=n2[-2]*val_j))
            plt.colorbar()
            plt.savefig(folder_name+"phasematching function "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power_avg_in)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'.png', format='png')
            #plt.show()

            plt.figure(figure_num+3)
            plt.title("JSA "+'pump scale: '+"{:.2f}".format(scale_p)+', signal scale: '+"{:.2f}".format(scale))
            plt.pcolor(omega_sig,omega_sig,f(E0,omega_sig[:,None],omega_sig[None,:],omega_p,pump_duration,leff(val_i,val_j),fo1=n1[-2]*val_i, fo2=n2[-2]*val_j))
            plt.colorbar()
            plt.savefig(folder_name+"JSA "+'pump scale '+"{:.2f}".format(scale_p)+', signal scale '+"{:.2f}".format(scale)+'_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(power_avg_in)+'W_'+"{:.2f}".format(scale_p)+'_'+"{:.2f}".format(scale)+'_'+str(wl)+'.png', format='png')
            figure_num+=4
            #plt.show()
            p.clear()
        #print("--- %s seconds ---" % (time.time() - start_time))

    vg_p=1/(arr*n1[-2]*omega_p/c+n1[-1]/c)
    vg=1/(arr*n2[-2]*omega_p/2/c+n2[-1]/c)

    # plt.figure(10000)
    # plt.pcolor(gains)
    # plt.colorbar()
    # #plt.show()
    # plt.savefig('First mode gain_'+str(num_freq_pts)+'_' + str(num_z_pts)+'.png', format='png')
    #
    # plt.figure(10001)
    # plt.pcolor(schmidt_n)
    # plt.colorbar()
    # #plt.show()
    # plt.savefig('Schmidt number_'+str(num_freq_pts)+'_' + str(num_z_pts)+'.png', format='png')

    print("--- %s seconds ---" % (time.time() - start_time))
    # to_save = {'arr':arr ,'gains':gains, 'schmidt_n':schmidt_n, 'vg':vg, 'vg_p':vg_p, 'domg':domg,'ttime':(time.time() - start_time)}
    # file_name = 'Green_Functions_'+str(num_freq_pts)+'_' + str(num_z_pts)+'_'+str(pump_duration*1e12)+"ps_"+str(power_avg_in)+"W_gains"
    # if part>0:
    #     file_name+='_part_'+str(part)
    # with open(file_name+'.pickle','wb') as handle:
    #     pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


#for omega_index,omega in enumerate(omega_sig):
#    a[0,omega_index] = 1/domg
#    start_time2 = time.time()
#    a_out = z_propagation(omega_sig, a,lp,len(lp),k_sig,k_pump,E_pump)
#    print("--- this step took %s seconds ---" % (time.time() - start_time2))
#    a_out[-1, omega_index] = 0
    #print('Done Step 1')
#    sum[omega_index,:] = a_out[-1,:]
#    a[0,omega_index] = 1j/domg
#    a_out = z_propagation(omega_sig, a,lp,len(lp),k_sig,k_pump,E_pump)
#    a_out[-1, omega_index] = 0
#    diff[omega_index,:] = a_out[-1,:]
#    a[0,omega_index] = 0
#    test = (sum[omega_index,:] + diff[omega_index,:])/(1 + 1j)

## TODO
#1.) Calculate S and C wtih appropriate phase adjustments (Switch back from Interaction Picture)
#2.) Save S,C, omega_sig, E0, omega_p_c, pump_duration as a pickle file.
#3.) Do modal Decomposition
#S = (sum+diff)/(1+1j)
#G = (sum-diff)/(1+1j)
