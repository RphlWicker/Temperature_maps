from astropy.io import fits
import numpy as np

print('Select your cluster :')
selected_clust = input()
if selected_clust == '282':
    file_root = 'CL0282_snap107_sph_smoothed'
    header = fits.getheader('/home2/ferragamo/NIKA2_300_TS/Raphael_maps/Xray_theoretical_maps_CL0282_snap_107_proj_0.fits', ext=1)
elif selected_clust == '9':
    file_root = 'CL0009_snap104_sph_smoothed'
    header = fits.getheader('/home2/ferragamo/NIKA2_300_TS/Raphael_maps/Xray_theoretical_maps_CL0009_snap_104_proj_0.fits', ext=1)

print(header)

temperature_volume = np.load(f'{file_root}_temp_map_4r200.npy')
metallicity_volume = np.load(f'{file_root}_Z_map_4r200.npy')
particle_volume = np.load(f'{file_root}_parts_map_4r200.npy')
density_volume = np.load(f'{file_root}_ne_map_4r200.npy')

mean_temperature_volume = temperature_volume/particle_volume
#mean_metallicity_volume = metallicity_volume/particle_volume

temperature_map = np.sum(mean_temperature_volume*particle_volume, axis=2)/np.sum(particle_volume, axis=2)
metallicity_map = np.sum(metallicity_volume*particle_volume, axis=2)/np.sum(particle_volume, axis=2)
squared_density_map = np.sum(density_volume**2, axis=2)/np.sum(particle_volume, axis=2)

hdulist = fits.HDUList()
hdulist.append(fits.PrimaryHDU())
header['EXTNAME'] = 'TEMPERATURE'
hdulist.append(fits.ImageHDU(data=temperature_map, header=header, name='Temperature', do_not_scale_image_data=True))
header['EXTNAME'] = 'Z'
hdulist.append(fits.ImageHDU(data=metallicity_map, header=header, name='Metallicity', do_not_scale_image_data=True))
header['EXTNAME'] = 'NE_2'
hdulist.append(fits.ImageHDU(data=squared_density_map, header=header, name='Squared Density', do_not_scale_image_data=True))

hdulist.writeto(f'{file_root}_emission_map_kit.fits', overwrite=True)