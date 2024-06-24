# %%
import numpy as np
import pandas as pd
import h5py
# from readsnapsgl import readsnapsgl
from astropy import constants as const
from astropy import units
from astropy.io import fits
from scipy.spatial import cKDTree
import os, sys
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import readsnapsgl
from astropy.cosmology import FlatLambdaCDM
from utils import rotate_data
from tqdm import tqdm

# %%
mean_mol_weight = 0.588
prtn = 1.67373522381e-24  # (proton mass in g)
bk = 1.3806488e-16        # (Boltzman constant in CGS)

cosm = FlatLambdaCDM(H0=67.77, Om0=0.307115)

# %%
#=======================================================
#simulation = 'Gizmo-Simba'
simulation = 'GadgetX'

NIKA2_TS_df = pd.read_csv(f'/home2/ferragamo/NIKA2_300_TS/NIKA2-TS_clusters_list.csv')
NIKA2_morpho_par_df = pd.read_csv('/data7/NIKA2-300th/Data_Base/Nika2_twin_sample_chi_par.csv')
NIKA2_TS_df = NIKA2_TS_df.merge(NIKA2_morpho_par_df[['rid', 'hid', 'Chi500_DL', 'Chi500_H',  'Chi200_DL', 'Chi200_H']], how='inner', on=['rid', 'hid'])
for i, (crn, sn, hid, name, ts, chi200_dl, chi200_h, chi500_dl, chi500_h) in enumerate(tqdm(zip(NIKA2_TS_df.rid, NIKA2_TS_df.snap, NIKA2_TS_df.hid, 
                                                                                                NIKA2_TS_df.LPSZ_ShortName, NIKA2_TS_df.TS,
                                                                                                NIKA2_TS_df.Chi200_DL, NIKA2_TS_df.Chi200_H,
                                                                                                NIKA2_TS_df.Chi500_DL, NIKA2_TS_df.Chi500_H))):
    cn = 'NewMDCLUSTER_%04d'%crn

    snapn='snap_%03d'%sn

    if simulation == 'Gizmo-Simba':
        Xspath = "/home2/weiguang/data7/Gizmo-Simba/"
        filename = Xspath+cn+'/'+snapn+'.hdf5'
        progenIDs = np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GIZMO/Progenitor-IDs-for-center-cluster.txt",dtype=np.int64)
        tmpd = np.load('/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GIZMO/GS_Mass_snap_%03dinfo.npy' %sn)
    elif simulation == 'GadgetX':
        Xspath = '/data4/niftydata/TheThreeHundred/data/simulations/GadgetX/'
        filename = Xspath+cn+'/'+snapn
        progenIDs = np.loadtxt("/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/Progenitor-IDs-for-center-cluster.txt",dtype=np.int64)
        tmpd = np.load('/home2/weiguang/Project-300-Clusters/Halo_mass_function_mass-difference/GadgetX/G3X_Mass_snap_%03dinfo.npy' %sn)

    sn_sa_red = np.loadtxt('/home2/weiguang/Project-300-Clusters/redshifts.txt')

    # %%
    if simulation == 'Gizmo-Simba':
        f = h5py.File(filename, 'r')
        v_unit = 1.0e5 * np.sqrt(f['Header'].attrs['Time'])  # (e.g. 1.0 km/sec)
    elif simulation == 'GadgetX':
        head = readsnapsgl.readsnap(filename, block = 'HEAD', quiet=True, ptype=0)
        v_unit = 1.0e5 * np.sqrt(head.Time)  # (e.g. 1.0 km/sec)

    m_ptoMsun = const.m_p.to('Msun').value #proton mass in unit Msun
    KtokeV = const.k_B.value / const.e.value * 1e-3 # [K] to [keV] == k_B*T
    kinunit = 1/ units.keV.to('Msun km2 s-2') #from Msun*(km/s)^2 to keV

    # %%
    #ReginIDs HIDs  HosthaloID Mvir(4) Xc(5)   Yc(6)   Zc(7)  Rvir(8) fMhires(38) cNFW (42) Mgas200 M*200 M500(13)  R500(14) fgas500 f*500
    ids = np.where((np.int32(tmpd[:,0])==crn) & (np.int64(tmpd[:,1]) == progenIDs[crn-1, sn]))[0]

    cc = tmpd[ids, 4:7][0]
    r200 = tmpd[ids, 7][0]
    r500 = tmpd[ids, 13][0]
    M500 = tmpd[ids, 12][0]*1.e10
    M200 = tmpd[ids,3][0]
    #print(M200, r200, cc)

    prop = { 'cc':cc,'r200':r200}

    # %%
    j = 0 #particle type: 0: gas; 1: DM; 4: stars

    if simulation == 'Gizmo-Simba':
        vel_g = f['PartType'+str(j)+'/Velocities'][:]
        pos_g = f['PartType'+str(j)+'/Coordinates'][:]
        rr_g = np.sqrt(np.sum((pos_g-prop['cc'])**2, axis=1))
        mas_g = f['PartType'+str(j)+'/Masses'][:]
        sgden_g = f['PartType'+str(j)+'/Density'][:] * 1.0e10 /0.6777/(1/0.6777)**3  #in Msun/kpc^3
        # temp= f['PartType'+str(j)+'/InternalEnergy'][:] * (5. / 3 - 1) * v_unit**2 * prtn * mean_mol_weight / bk # in k
        temp_g = readsnapsgl.readhdf5data(filename, block = 'temperature', quiet=True, ptype=j) #in K
        eleab_g = f['PartType'+str(j)+'/ElectronAbundance'][:]
        #wind= f['PartType'+str(j)+'/NWindLaunches'][:]
        delaytime_g = f['PartType'+str(j)+'/DelayTime'][:]
        sfr_g = f['PartType'+str(j)+'/StarFormationRate'][:]
        metl_g = f['PartType'+str(j)+'/Metallicity'][:]
        pot_g = f['PartType'+str(j)+'/Potential'][:]
        metHe_g = metl_g[:,1] #He 
        metM_g = metl_g[:,0] #metallicity
        nH_g = (1 - metHe_g - metM_g)*mas_g*1.0e10 / 0.6777  #proton number 
        ne_g = nH_g * eleab_g #electron number: proton number multiply relative value

    elif simulation == 'GadgetX':
        vel_g = readsnapsgl.readsnap(filename, block = 'VEL ', quiet=True, ptype=j)
        pos_g = readsnapsgl.readsnap(filename, block = 'POS ', quiet=True, ptype=j)
        rr_g = np.sqrt(np.sum((pos_g-prop['cc'])**2, axis=1))
        mas_g = readsnapsgl.readsnap(filename, block = 'MASS', quiet=True, ptype=j)
        sgden_g = readsnapsgl.readsnap(filename, block = 'RHO ', quiet=True, ptype=j) * 1.0e10 /0.6777/(1/0.6777)**3  #in Msun/kpc^3
        temp_g = readsnapsgl.readsnap(filename, block = 'TEMP', quiet=True, ptype=j)
        eleab_g = readsnapsgl.readsnap(filename, block = 'NE  ', quiet=True, ptype=j)
        sfr_g = readsnapsgl.readsnap(filename, block = 'SFR ', quiet=True, ptype=j)
        metHe_g = 0.24 #He 
        metM_g = readsnapsgl.readsnap(filename, block = 'Z   ', quiet=True, ptype=j) #total metallicity
        nH_g = (1 - metHe_g - metM_g)*mas_g*1.0e10 / 0.6777  #proton number 
        ne_g = nH_g * eleab_g #electron number: proton number multiply relative value       

    # %%(Figure Updated_samples.jpg)(Figure Updated_samples.jpg)
    #projection along 29 different los
    rotations = np.loadtxt('/home2/ferragamo/NIKA2_300_TS/29_rotations.txt')
    for irot, RA in enumerate(rotations):
        pos_g, vel_g, _= rotate_data(pos_g, RA, vel=vel_g, bvel=None)
        prop['cc'], _, _= rotate_data(prop['cc'], RA, vel=None, bvel=None)

        # %%
        pos_x = np.array([pos[0]-prop['cc'][0] for pos in pos_g])
        pos_y = np.array([pos[1]-prop['cc'][1] for pos in pos_g])
        pos_z = np.array([pos[2]-prop['cc'][2] for pos in pos_g])

        # %%
        r200_frac = 1.5

        # %%
        #particle conditions
        if simulation == 'Gizmo-Simba':
                ids0 = [(np.abs(pos_x) <= r200_frac*prop['r200']) & 
                        (np.abs(pos_y) <= r200_frac*prop['r200']) & 
                        (np.abs(pos_z) <= r200_frac*prop['r200']) &
                        (temp_g > 1.0e6) & 
                        (delaytime_g <= 0) &
                        (sgden_g < 2.88e6)][0]
        if simulation == 'GadgetX':
                ids0 = [(np.abs(pos_x) <= r200_frac*prop['r200']) & 
                        (np.abs(pos_y) <= r200_frac*prop['r200']) & 
                        (np.abs(pos_z) <= r200_frac*prop['r200']) &
                        (temp_g > 1.0e6) & 
                        (sgden_g < 2.88e6)][0]

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
        eleab = eleab_g[ids0]

        if simulation == 'Gizmo-Simba':
                pot = pot_g[ids0]

        kin_energy_gas = np.sum(vel**2,axis=1,dtype=np.float64)*mas/2./0.6777 #unit: Msun (km/s)^2
        them_energy_gas = 3.*KtokeV*mas / 0.6777 * temp / 2. #unit: keV


        # %%
        x = pos_x[ids0]#/prop['r200']
        y = pos_y[ids0]#/prop['r200']
        z = pos_z[ids0]#/prop['r200']

        # %%
        """
        hostid = hid
        fahf = glob.glob('/data4/niftydata/TheThreeHundred/data/catalogues/AHF/GIZMO/%s/GIZMO-%s.%s.z*.AHF_halos' %(cn,cn,snapn))
        ahf = np.loadtxt(fahf[0])
        idv = np.where(ahf[:,1] == hostid)[0]
        subhalo_v = ahf[idv,8:11]
        subhalo_x = ahf[idv,5:8]
        rrsubhalo = np.sqrt(np.sum((subhalo_x-prop['cc'])**2, axis=1))
        """
        # %%
        nbins = int(prop['r200']*r200_frac/10)*2#300
        rbins = np.linspace(-prop['r200']*r200_frac, prop['r200']*r200_frac, num=nbins+1)

        psize = np.abs(rbins[0]-rbins[1])

        # %%
        #bin_kp = np.abs((rbins [0]-rbins[1])*prop['r200'])
        vol = psize**2 * prop['r200']*r200_frac  #unit: (kpc/h)^3
        volcm = vol*const.kpc.to('cm').value**3 #unit: (cm/h)^3	

        # %%
        #mass weighted temperature
        h_temp_mass, xedges, yedges, = np.histogram2d(x, y, bins=[nbins,nbins], weights=temp*mas)
        h_temp_mass = h_temp_mass.transpose()

        h_mass, _, _, = np.histogram2d(x, y, bins=[xedges, yedges], weights=mas)
        h_mass = h_mass.transpose()

        with np.errstate(divide='ignore', invalid='ignore'):
             h_tm = h_temp_mass/h_mass
        h_tm[~np.isfinite(h_tm)] = 0.

        xx, yy = np.meshgrid(xedges, yedges)

        # %%
        #temperature
        h_t, _, _, = np.histogram2d(x, y, bins=[xedges, yedges], weights=temp)
        h_t = h_t.transpose()
        h_part, _, _, = np.histogram2d(x, y, bins=[xedges, yedges])
        h_part = h_part.transpose()

        with np.errstate(divide='ignore', invalid='ignore'):
             h_temp = h_t/h_part
        h_temp[~np.isfinite(h_temp)] = 0.

        # %%
        #electron
        h_ne, _, _, = np.histogram2d(x, y, bins=[xedges, yedges], weights=ne)
        h_ne = h_ne.transpose()

        h_ne = h_ne /volcm / m_ptoMsun #/ 0.6777
        h_ne2 = h_ne**2

        h_g = h_ne *(1+1/np.nanmin(eleab))
        h_g2 = h_g**2

        # %%
        #density
        h_dens = h_mass/vol

        # %%
        # pressure: P = \sum (k_B/\mu/m_p)\rho_i*T_i from Planelles+2017
        #unit: keV * (cm/h)^(-3)
        #mass-weighted
        h_pressure = h_temp_mass/volcm/0.6125/m_ptoMsun/0.6777

        # %%
        #spectroscopic like temperature
        w = sgden*mas*temp**(-3/4)
        h_temp_w, _ , _, = np.histogram2d(x, y, bins=[nbins,nbins], weights=temp*w)
        h_temp_w = h_temp_w.transpose()

        h_w, _, _, = np.histogram2d(x, y, bins=[xedges, yedges], weights=w)
        h_w = h_w.transpose()

        with np.errstate(divide='ignore', invalid='ignore'):
             h_spec_temp = h_temp_w/h_w
        h_spec_temp[~np.isfinite(h_spec_temp)] = 0.

        # %%
        hdr = fits.Header()
        hdr.update({'PSIZE': (psize, 'kpc'), 
                    'region': '%04d'%crn,
                    'halo_ID': hid, 
                    'Snap': '%03d'%sn, 
                    'Redshift': sn_sa_red[sn][2], 
                    'r200': (r200, 'h^-1 kpc'), 
                    'M200': (M200/1.e14, '10^14 h^-1 Msun'), 
                    'r500': (r500, 'h^-1 kpc'), 
                    'M500': (M500/1.e14, '10^14 h^-1 Msun'),
                    'Proj': f'{RA[0]}, {RA[1]}, {RA[2]}',
                    'Chi200DL': chi200_dl,
                    'Chi200H': chi200_h,
                    'Chi500DL': chi500_dl,
                    'Chi500H': chi500_h,
                    'LP-NAME': name,
                    'TS': ts
                    })

        # %%
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(data=h_mass, name='Mass', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_temp, name='Temperature', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_tm, name='Mass weighted Temperature', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_spec_temp, name='Spectroscopic like Temperature', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_pressure, name='Pressure', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_ne, name='electron density', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_ne2, name='square electron number density', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_g, name='Gas number Density', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_g2, name='square Gas number Density', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_t, name='T', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_part, name='n_p', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_temp_mass, name='Tm', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_temp_w, name='Tw', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(data=h_w, name='w', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(xedges, name='X', header=hdr, do_not_scale_image_data=True))
        hdul.append(fits.ImageHDU(yedges, name='Y', header=hdr, do_not_scale_image_data=True))


        for hdu, map_units in zip(hdul[1:], ['Msun/h', 'Msun/h/kpc^3', 'keV', 'keV', 'keV', 'keV * (cm/h)^(-3)', 'n/(h^-1cm)^3',
                                             'keV', 'n', 'keV*H^-1Msun', '(h^-1Msun)*2keV^-1/4', '(h^-1Msun)*2keV^-3/4', 'kpc', 'kpc']):
            hdu.header.comments['EXTNAME'] = map_units

        map_path = f'/home2/ferragamo/NIKA2_300_TS/Raphael_maps/'
        #map_file = f'Xray_theoretical_maps_CL{crn :04d}_snap_{sn :03d}.fits'
        map_file = f'Xray_theoretical_maps_CL{crn :04d}_snap_{sn :03d}_proj_{irot}.fits'
        hdul.writeto(f'{map_path}{map_file}', overwrite=True)
