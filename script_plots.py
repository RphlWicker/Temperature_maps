import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import os
import sys
import glob
import readsnapsgl
from utils import rotate_data
from pymsz.rotate_data import SPH_smoothing
from time import time

mean_mol_weight = 0.588
prtn = 1.67373522381e-24  # (proton mass in g)
bk = 1.3806488e-16        # (Boltzman constant in CGS)

cosm = FlatLambdaCDM(H0=67.77, Om0=0.307115)

NIKA2_TS_df = pd.read_csv(f'/home2/ferragamo/NIKA2_300_TS/NIKA2-TS_clusters_list.csv')
NIKA2_morpho_par_df = pd.read_csv('/data7/NIKA2-300th/Data_Base/Nika2_twin_sample_chi_par.csv')
NIKA2_TS_df = NIKA2_TS_df.merge(NIKA2_morpho_par_df[['rid', 'hid', 'Chi500_DL', 'Chi500_H',  'Chi200_DL', 'Chi200_H']], how='inner', on=['rid', 'hid'])

for i, (crn, sn, hid, name, ts, chi200_dl, chi200_h, chi500_dl, chi500_h) in enumerate(zip(NIKA2_TS_df.rid, NIKA2_TS_df.snap, NIKA2_TS_df.hid, 
                                                                                           NIKA2_TS_df.LPSZ_ShortName, NIKA2_TS_df.TS,
                                                                                           NIKA2_TS_df.Chi200_DL, NIKA2_TS_df.Chi200_H,
                                                                                           NIKA2_TS_df.Chi500_DL, NIKA2_TS_df.Chi500_H)):
    print(crn, sn, hid,name, ts, chi200_dl, chi200_h, chi500_dl, chi500_h)
    break


crn = 282 #9
cn = f'NewMDCLUSTER_{crn:04d}'

sn = 107 #104
snapn = f'snap_{sn:03d}'

#=======================================================
#simulation = 'Gizmo-Simba'
simulation = 'GadgetX'

if simulation == 'Gizmo-Simba':
    Xspath = "/home2/weiguang/data7/Gizmo-Simba/"
    filename = f'{Xspath}{cn}/{snapn}.hdf5'
    progenIDs = np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GIZMO/Progenitor-IDs-for-center-cluster.txt",dtype=np.int64)
    tmpd = np.load(f'/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GIZMO/GS_Mass_snap_{sn:03d}info.npy')
elif simulation == 'GadgetX':
    Xspath = '/data4/niftydata/TheThreeHundred/data/simulations/GadgetX/'
    filename = f'{Xspath}{cn}/{snapn}'
    progenIDs = np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/Progenitor-IDs-for-center-cluster.txt",dtype=np.int64)
    tmpd = np.load(f'/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/G3X_Mass_snap_{sn:03d}info.npy')

sn_sa_red = np.loadtxt('/home2/weiguang/Project-300-Clusters/redshifts.txt')

sn_sa_red = np.loadtxt('/home2/weiguang/Project-300-Clusters/redshifts.txt')
sn_sa_red
kk = np.array([i[2] for i in sn_sa_red])

if simulation == 'Gizmo-Simba':
    file_data = h5py.File(filename, 'r')
    v_unit = 1.0e5 * np.sqrt(f['Header'].attrs['Time'])  # (e.g. 1.0 km/sec)
elif simulation == 'GadgetX':
    head = readsnapsgl.readsnap(filename, block = 'HEAD', quiet=True, ptype=0)
    v_unit = 1.0e5 * np.sqrt(head.Time)  # (e.g. 1.0 km/sec)

m_ptoMsun = const.m_p.to('Msun').value #proton mass in unit Msun
KtokeV = const.k_B.value / const.e.value * 1e-3 # [K] to [keV] == k_B*T
kinunit = 1/ u.keV.to('Msun km2 s-2') #from Msun*(km/s)^2 to keV

#ReginIDs HIDs  HosthaloID Mvir(4) Xc(5)   Yc(6)   Zc(7)  Rvir(8) fMhires(38) cNFW (42) Mgas200 M*200 M500(13)  R500(14) fgas500 f*500
ids = np.where((np.int32(tmpd[:,0])==crn) & (np.int64(tmpd[:,1]) == progenIDs[crn-1, sn]))[0]

cc = tmpd[ids, 4:7][0]
r200 = tmpd[ids, 7][0]
r500 = tmpd[ids, 13][0]
M500 = tmpd[ids, 12][0]*1.e10
M200 = tmpd[ids,3][0]
#print(M200, r200, cc)

prop = { 'cc':cc,'r200':r200}

j = 0 #particle type: 0: gas; 1: DM; 4: stars

if simulation == 'Gizmo-Simba':
    vel_g = file_data[f'PartType{j}/Velocities'][:]
    pos_g = file_data[f'PartType{j}/Coordinates'][:]
    #rr_g = np.sqrt(np.sum((pos_g-prop['cc'])**2, axis=1))
    mas_g = file_data[f'PartType{j}/Masses'][:]
    sgden_g = file_data[f'PartType{j}/Density'][:] * 1.0e10 /0.6777/(1/0.6777)**3  #in Msun/kpc^3
    # temp = file_data[f'PartType{j}/InternalEnergy'][:] * (5. / 3 - 1) * v_unit**2 * prtn * mean_mol_weight / bk # in k
    temp_g = readsnapsgl.readhdf5data(filename, block = 'temperature', quiet=True, ptype=j) #in K
    eleab_g = file_data[f'PartType{j}/ElectronAbundance'][:]
    #wind = file_data[f'PartType{j}/NWindLaunches'][:]
    delaytime_g = f[f'PartType{j}/DelayTime'][:]
    sfr_g = file_data[f'PartType{j}/StarFormationRate'][:]
    metl_g = file_data[f'PartType{j}/Metallicity'][:]
    pot_g = file_data[f'PartType{j}/Potential'][:]
    metHe_g = metl_g[:,1] #He 
    metM_g = metl_g[:,0] #metallicity
    nH_g = (1 - metHe_g - metM_g)*mas_g*1.0e10 / 0.6777  #proton number 
    ne_g = nH_g * eleab_g #electron number: proton number multiply relative value

elif simulation == 'GadgetX':
    vel_g = readsnapsgl.readsnap(filename, block = 'VEL ', quiet=True, ptype=j)
    pos_g = readsnapsgl.readsnap(filename, block = 'POS ', quiet=True, ptype=j)
    #rr_g = np.sqrt(np.sum((pos_g-prop['cc'])**2, axis=1))
    mas_g = readsnapsgl.readsnap(filename, block = 'MASS', quiet=True, ptype=j)
    sgden_g = readsnapsgl.readsnap(filename, block = 'RHO ', quiet=True, ptype=j) * 1.0e10 /0.6777/(1/0.6777)**3  #in Msun/kpc^3
    temp_g = readsnapsgl.readsnap(filename, block = 'TEMP', quiet=True, ptype=j)
    eleab_g = readsnapsgl.readsnap(filename, block = 'NE  ', quiet=True, ptype=j)
    sfr_g = readsnapsgl.readsnap(filename, block = 'SFR ', quiet=True, ptype=j)
    metHe_g = 0.24 #He 
    metM_g = readsnapsgl.readsnap(filename, block = 'Z   ', quiet=True, ptype=j) #total metallicity
    nH_g = (1 - metHe_g - metM_g)*mas_g*1.0e10 / 0.6777  #proton number 
    ne_g = nH_g * eleab_g #electron number: proton number multiply relative value
    smoothing_length = readsnapsgl.readsnap(filename, block='HSML', quiet=True, ptype=j)

rott = np.loadtxt('/home2/ferragamo/NIKA2_300_TS/29_rotations.txt')
for irot, RA in enumerate(rott):
    RA = rott[0]
    pos_g, vel_g, _= rotate_data(pos_g, RA, vel=vel_g, bvel=None)
    prop['cc'], _, _= rotate_data(prop['cc'], RA, vel=None, bvel=None)
    rr_g = np.sqrt(np.sum((pos_g-prop['cc'])**2, axis=1))
    break

pos_x = np.array([pos[0]-prop['cc'][0] for pos in pos_g])
pos_y = np.array([pos[1]-prop['cc'][1] for pos in pos_g])
pos_z = np.array([pos[2]-prop['cc'][2] for pos in pos_g])

r200_frac = 3.5
pxls = 20

#particle conditions
temp_limit = 1.0e6 #5.8e6
dens_limit = 2.88e6
if simulation == 'Gizmo-Simba':
        ids0 = [(np.abs(pos_x) <= r200_frac*prop['r200']) & 
                (np.abs(pos_y) <= r200_frac*prop['r200']) & 
                (np.abs(pos_z) <= r200_frac*prop['r200']) &
                (temp_g > temp_limit) & 
                (delaytime_g <= 0) &
                (sgden_g < dens_limit)][0]
if simulation == 'GadgetX':
        ids0 = [(np.abs(pos_x) <= r200_frac*prop['r200']) & 
                (np.abs(pos_y) <= r200_frac*prop['r200']) & 
                (np.abs(pos_z) <= r200_frac*prop['r200']) &
                (temp_g > temp_limit) & 
                (sgden_g < dens_limit)][0]

rr = rr_g[ids0]
mas = mas_g[ids0]*1.0e10
temp = temp_g[ids0]
temp = temp*KtokeV
vel = vel_g[ids0]
sgden = sgden_g[ids0]
hubblev = pos_g[ids0] / 0.6777 * 67.77 / 1e3 #Hubble flow; unit: km/s
vv = vel + hubblev
ne = ne_g[ids0]
metM = metM_g[ids0]
smoothing_length = smoothing_length[ids0]

if simulation == 'Gizmo-Simba':
        pot = pot_g[ids0]
        
kin_energy_gas = np.sum(vel**2,axis=1,dtype=np.float64)*mas/2./0.6777 #unit: Msun (km/s)^2
them_energy_gas = 3.*KtokeV*mas / 0.6777 * temp / 2. #unit: keV

x = pos_x[ids0]
y = pos_y[ids0]
z = pos_z[ids0]

if simulation == 'Gizmo-Simba':
        ids0_p = [(rr_g <= r200_frac*prop['r200']) & 
                (temp_g > temp_limit) & 
                (delaytime_g <= 0) &
                (sgden_g < dens_limit)][0]
if simulation == 'GadgetX':
        ids0_p = [(rr_g <= r200_frac*prop['r200']) & 
                (temp_g > temp_limit) & 
                (sgden_g < dens_limit)][0]

rr_p = rr_g[ids0_p]
ne_p = ne_g[ids0_p]

x_p = pos_x[ids0_p]
y_p = pos_y[ids0_p]
z_p = pos_z[ids0_p]

nbins = int(prop['r200']*r200_frac/pxls)*2
rbins = np.linspace(-prop['r200']*r200_frac, prop['r200']*r200_frac, num=nbins+1)

psize = np.abs(rbins[0]-rbins[1])

bin_kp = np.abs((rbins [0]-rbins[1]))#*prop['r200'])
bin_cm = bin_kp*const.kpc.to('cm').value #unit: (cm/h)^3	

vol = bin_kp**3  #unit: (kpc/h)^3
volcm = vol*const.kpc.to('cm').value**3 #unit: (cm/h)^3	

ne_sph = np.load('test_sph_smoothed_ne_map_7.0r200.npy')
summed_ne = np.sum(ne_sph,axis=2)*bin_kp
summed_ne2 = np.sum(ne_sph**2,axis=2)*bin_kp

leff_map = summed_ne**2/summed_ne2

psize = 2*r200_frac*r200/len(leff_map)
print(psize)
r500_pix = r500/psize
r200_pix = r200/psize

distance_grid_x, distance_grid_y = np.meshgrid(np.linspace(-len(leff_map)*psize//2, len(leff_map)*psize//2, len(leff_map)), 
                                               np.linspace(-len(leff_map)*psize//2, len(leff_map)*psize//2, len(leff_map)))

arcmin_grid_x = distance_grid_x / cosm.kpc_proper_per_arcmin(0.592).value
arcmin_grid_y = distance_grid_y / cosm.kpc_proper_per_arcmin(0.592).value
r500_arcmin = r500/cosm.kpc_proper_per_arcmin(0.592).value
r200_arcmin = r200/cosm.kpc_proper_per_arcmin(0.592).value
radius_1arcmin = cosm.kpc_proper_per_arcmin(0.592).value

def plot_map():
    fig =plt.figure(figsize=(10,10))
    ax1=fig.add_subplot(111, aspect='equal')
    test_leffmap = plt.pcolormesh(distance_grid_x, distance_grid_y, leff_map, cmap='Spectral_r')
    circle_R500 = plt.Circle((0., 0.),
                            r500, color='k', fill=False, lw=3)
    circle_R200 = plt.Circle((0., 0.),
                            r200, color='k', fill=False, lw=3)
    circle_1arcmin = plt.Circle((0., 0.),
                            radius_1arcmin, color='k', fill=False, lw=3, ls='--')
    ax1.add_artist(circle_R200)
    ax1.add_artist(circle_R500)
    ax1.add_artist(circle_1arcmin)
    ax1.set_title(r'$l_{eff} = \frac{(\int n_e dl)^2}{\int n_e^2 dl}$')
    plt.colorbar(test_leffmap)

    plt.savefig(f'SPH_leff_pix{int(psize)}kpc_{int(2*r200_frac)}r200.jpg')
    plt.show()

def plot_profile():
    nbin = 150
    rbins_p = np.logspace(np.log10(0.005*prop['r200']), np.log10(r200_frac*prop['r200']), num=nbin+1)

    xx, yy = np.meshgrid(np.linspace(-len(leff_map)//2, len(leff_map)//2, len(leff_map)), 
                         np.linspace(-len(leff_map)//2, len(leff_map)//2, len(leff_map)))
    dist = np.sqrt(xx**2 + yy**2)
    
    leff_prof = []
    leff_prof.append(np.nanmean(leff_map[(dist<=rr[0])]))
    for i, (rr0, rr1) in enumerate(zip(rbins_p[:-1], rbins_p[1:])):
        leff_prof = np.append(leff_prof, np.nanmean(leff_map[(dist>rr0) & (dist<=rr1)]))
    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    ax.plot(rbins_p, leff_prof)
    ax.vlines([radius_1arcmin, r500, r200], 1., 1e4, colors='k', ls='--', lw=2)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('$l_{eff}$ [kpc]')
    plt.savefig('leff_profile_sph.jpg')
    plt.show()

