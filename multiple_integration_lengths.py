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
from time import time

mean_mol_weight = 0.588
prtn = 1.67373522381e-24  # (proton mass in g)
bk = 1.3806488e-16        # (Boltzman constant in CGS)

cosm = FlatLambdaCDM(H0=67.77, Om0=0.307115)

NIKA2_TS_df = pd.read_csv(f'/home2/ferragamo/NIKA2_300_TS/NIKA2-TS_clusters_list.csv')
NIKA2_morpho_par_df = pd.read_csv('/data7/NIKA2-300th/Data_Base/Nika2_twin_sample_chi_par.csv')
NIKA2_TS_df = NIKA2_TS_df.merge(NIKA2_morpho_par_df[['rid', 'hid', 'Chi500_DL', 'Chi500_H',  'Chi200_DL', 'Chi200_H']], how='inner', on=['rid', 'hid'])

crn = 282 #9
cn = f'NewMDCLUSTER_{crn:04d}'

sn = 107 #104
snapn = f'snap_{sn:03d}'

Xspath = '/data4/niftydata/TheThreeHundred/data/simulations/GadgetX/'
filename = f'{Xspath}{cn}/{snapn}'
progenIDs = np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/Progenitor-IDs-for-center-cluster.txt",dtype=np.int64)
tmpd = np.load(f'/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/G3X_Mass_snap_{sn:03d}info.npy')

sn_sa_red = np.loadtxt('/home2/weiguang/Project-300-Clusters/redshifts.txt')
sn_sa_red
kk = np.array([i[2] for i in sn_sa_red])

head = readsnapsgl.readsnap(filename, block = 'HEAD', quiet=True, ptype=0)
v_unit = 1.0e5 * np.sqrt(head.Time)  # (e.g. 1.0 km/sec)

m_ptoMsun = const.m_p.to('Msun').value #proton mass in unit Msun
KtokeV = const.k_B.value / const.e.value * 1e-3 # [K] to [keV] == k_B*T
kinunit = 1/ u.keV.to('Msun km2 s-2') #from Msun*(km/s)^2 to keV

ids = np.where((np.int32(tmpd[:,0])==crn) & (np.int64(tmpd[:,1]) == progenIDs[crn-1, sn]))[0]

cc = tmpd[ids, 4:7][0]
r200 = tmpd[ids, 7][0]
r500 = tmpd[ids, 13][0]
M500 = tmpd[ids, 12][0]*1.e10
M200 = tmpd[ids,3][0]
#print(M200, r200, cc)

prop = { 'cc':cc,'r200':r200}
j = 0 

vel_g = readsnapsgl.readsnap(filename, block = 'VEL ', quiet=True, ptype=j)
pos_g = readsnapsgl.readsnap(filename, block = 'POS ', quiet=True, ptype=j)
mas_g = readsnapsgl.readsnap(filename, block = 'MASS', quiet=True, ptype=j)
sgden_g = readsnapsgl.readsnap(filename, block = 'RHO ', quiet=True, ptype=j) * 1.0e10 /0.6777/(1/0.6777)**3  #in Msun/kpc^3
temp_g = readsnapsgl.readsnap(filename, block = 'TEMP', quiet=True, ptype=j)
eleab_g = readsnapsgl.readsnap(filename, block = 'NE  ', quiet=True, ptype=j)
sfr_g = readsnapsgl.readsnap(filename, block = 'SFR ', quiet=True, ptype=j)
metHe_g = 0.24 #He 
metM_g = readsnapsgl.readsnap(filename, block = 'Z   ', quiet=True, ptype=j) #total metallicity
nH_g = (1 - metHe_g - metM_g)*mas_g*1.0e10 / 0.6777  #proton number 
ne_g = nH_g * eleab_g #electron number: proton number multiply relative value
#smoothing_length = readsnapsgl.readsnap(filename, block='HSML', quiet=True, ptype=j)

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

temp_limit = 1.0e6 #5.8e6
dens_limit = 2.88e6

integrations_lengths = np.logspace(np.log10(0.25), np.log10(5), 15)

leff_profiles = []
radii_arrays = []
Q2_profiles = []
t0 = time()
for counter, r200_frac in enumerate(integrations_lengths) :
    t1 = time()
    print(f'Iteration number {counter} starting, box length is {2*r200_frac:.3f} R200, {2*(r200_frac*1.4):.3f} R500, {2*r200_frac*r200/1e3:.3f} Mpc')
    print('Selecting the particles')
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
    kin_energy_gas = np.sum(vel**2,axis=1,dtype=np.float64)*mas/2./0.6777 #unit: Msun (km/s)^2
    them_energy_gas = 3.*KtokeV*mas / 0.6777 * temp / 2. #unit: keV

    x = pos_x[ids0]
    y = pos_y[ids0]
    z = pos_z[ids0]

    print('Building and smoothing the volume (R = 30kpc)')
    nbins = int(prop['r200']*r200_frac/10)*2
    rbins = np.linspace(-prop['r200']*r200_frac, prop['r200']*r200_frac, num=nbins+1)

    bin_kp = np.abs((rbins [0]-rbins[1]))#*prop['r200'])
    bin_cm = bin_kp*const.kpc.to('cm').value #unit: (cm/h)^3	

    vol = bin_kp**3  #unit: (kpc/h)^3
    volcm = vol*const.kpc.to('cm').value**3

    h_ne_grid, x_ne_grid = np.histogramdd([x,y,z], bins=(nbins, nbins, nbins), weights=np.float64(ne)/m_ptoMsun/0.6777)
    h_ne_grid = h_ne_grid.transpose()
    h_ne_3D = h_ne_grid/bin_cm**3

    h_ne_3D_smoothed = gaussian_filter(h_ne_3D, 3)

    print('Building leff profiles')

    nbin = 150
    rbins_p = np.logspace(np.log10(0.005*prop['r200']), np.log10(r200_frac*prop['r200']), num=nbin+1)

    xx, yy = np.meshgrid(x_ne_grid[0][:-1], x_ne_grid[1][:-1])
    dist = np.sqrt(xx**2 + yy**2)

    ne_3D = np.sum(h_ne_3D_smoothed, axis=2)*bin_kp
    ne2_3D = np.sum(h_ne_3D_smoothed**2, axis=2)*bin_kp

    Q2_map = np.divide(np.sum(h_ne_3D_smoothed**2, axis=2)/nbins, np.sum(h_ne_3D_smoothed, axis=2)**2/nbins)

    leff_map = np.divide(ne_3D**2, ne2_3D)

    leff_prof = []
    Q2_prof = []

    leff_prof.append(np.nanmean(leff_map[(dist<=rr[0])]))
    Q2_prof.append(np.nanmean(Q2_map[(dist<=rr[0])]))

    for i, (rr0, rr1) in enumerate(zip(rbins_p[:-1], rbins_p[1:])):
        leff_prof = np.append(leff_prof, np.nanmean(leff_map[(dist>rr0) & (dist<=rr1)]))
        Q2_prof = np.append(Q2_prof, np.nanmean(Q2_map[(dist>rr0) & (dist<=rr1)])) 
    leff_profiles.append(leff_prof)
    radii_arrays.append(rbins_p)
    Q2_profiles.append(Q2_prof)
    t2 = time()
    print(f'Iteration number {counter} for {2*r200_frac:.3f} R200 finished in {int((t2-t1)//60)} minutes and {int((t2-t1)%60)} seconds.')
np.save('leff_profiles_0_5_to_10_R200.npy', np.array(leff_profiles))
np.save('radial_bins_0_5_to_10_R200.npy', np.array(radii_arrays))
np.save('Q2_profiles_0_5_to_10_R200.npy', np.array(Q2_profiles))
t3 = time()
print(f'Done ! Task completed in {int((t3-t0)//3600)} hours, {int(((t3-t0)%3600)//60)} minutes and {int(((t3-t0)%3600)%60)} seconds')

    