from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import colors
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
from astropy.cosmology import Planck18

plt.rcParams.update({'font.size': 26})

sz_data_dir = '/cluster/data/NIKA2/Twin_Sample/NIKA2_mock_maps/NIKA2_like_maps_wo_noise/snap_'
xray_data_dir = '/cluster/data/CHEX-MATE/XMM_like_maps/XMM_like_maps/'
thermo_quantity_maps = '/home/cluster/antonio.ferragamo/data/NIKA2/Theoretical_X_ray_maps/Raphael_maps/'
nika2_resolution_maps = '/home/cluster/raphael.wicker/data/NIKA2_resolution_maps/'


def resample_sz_map(snap, sz_map, plot=False):
    data = fits.getdata(f'{sz_data_dir}{snap}/{sz_map}', ext=1)
    z = fits.getheader(f'{sz_data_dir}{snap}/{sz_map}', ext=1)['REDSHIFT']
    x = np.linspace(0, 197, 197)
    y = np.linspace(0, 197, 197)

    f = interp2d(x, y, data, kind='cubic')

    x2 = np.linspace(0, 197, 1200)
    y2 = np.linspace(0, 197, 1200)
    resampled_map = f(x2, y2)

    if plot:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.imshow(data)  # , norm=colors.LogNorm())

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.imshow(resampled_map)  # , norm=colors.LogNorm())

        print(data.shape, resampled_map.shape)
        plt.show()
    return resampled_map, z


def convolve_x_map(cluster, x_map, plot=False):
    x = np.linspace(0, 1200, 1200)
    y = np.linspace(0, 1200, 1200)
    X, Y = np.meshgrid(x, y)

    pixel_size = 0.000625833066667  # in degrees
    nika2_angres = 0.00305556  # in degrees
    nika2_pixres = nika2_angres / pixel_size

    data = fits.getdata(f'{xray_data_dir}{cluster}/{x_map}')
    convolved_map = gaussian_filter(data, nika2_pixres)
    print(convolved_map, convolved_map.shape)

    if plot:
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.imshow(data)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.imshow(convolved_map)

        plt.show()
    return convolved_map


def mock_observed_temperature_map(sz_snap, sz_map, xray_clust, xray_map):
    sz_data, z = resample_sz_map(sz_snap, sz_map)
    xray_data = convolve_x_map(xray_clust, xray_map)

    leff = 1600.
    mec2sig_T = 1.
    temperature_map = 1 / np.sqrt(leff) * mec2sig_T * np.sqrt(sz_data / (4 * np.pi * (1 + z) ** 4 * xray_data))

    print(temperature_map)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.imshow(temperature_map)
    ax1.contour(temperature_map, levels=np.logspace(-6, -2, 15))

    plt.show()
    return temperature_map


def temperature_simple_reconstruction(cluster, snap, proj, plot=False, convolved=False):
    if not convolved:
        mass_map = fits.getdata(f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits',
                                ext=1)
        density_map = fits.getdata(
            f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits',
            ext=7) * (1 + 1 / 1.1568)
        pressure_map = fits.getdata(
            f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits',
            ext=6)
    else:
        mass_map = fits.getdata(f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits',
                                ext=1)
        density_map = fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits',
            ext=7)
        pressure_map = fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits',
            ext=6)

    if not convolved:
        density_map[density_map == 0.] = np.nan
    reconstructed_temp = np.divide(pressure_map, density_map)
    if not convolved:
        reconstructed_temp = np.nan_to_num(reconstructed_temp)
        density_map = np.nan_to_num(density_map)

    if plot:
        fig1 = plt.figure(figsize=(25, 15))

        ax1 = fig1.add_subplot(121, aspect='equal')
        ne_map = ax1.pcolormesh(density_map)  # , norm=colors.LogNorm())
        plt.colorbar(ne_map)
        ax1.set_title('Density $n_e$')

        ax2 = fig1.add_subplot(122, aspect='equal')
        P_map = ax2.pcolormesh(pressure_map)  # , norm=colors.LogNorm())
        plt.colorbar(P_map)
        ax2.set_title('Pressure $P_e$')

        fig2 = plt.figure(figsize=(15, 15))

        ax3 = fig2.add_subplot(111, aspect='equal')
        T_map = ax3.pcolormesh(reconstructed_temp)  # , norm=colors.LogNorm())
        plt.colorbar(T_map)
        ax3.set_title('Reconstructed temperature')

        plt.show()

    return reconstructed_temp


def temperature_map_comparison(cluster, snap, proj, convolved=False):
    if not convolved:
        theoretical_temperature = fits.getdata(
            f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)
        mass_weighted_temp = fits.getdata(
            f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=4)
        spectro_like_temp = fits.getdata(
            f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=5)
        reconstructed_temp = temperature_simple_reconstruction(cluster, snap, proj)

        theoretical_temperature[theoretical_temperature == 0.] = np.nan
        mass_weighted_temp[mass_weighted_temp == 0.] = np.nan
        spectro_like_temp[spectro_like_temp == 0.] = np.nan

    else:
        theoretical_temperature = fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits',
            ext=8) / fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=9)
        mass_weighted_temp = fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits',
            ext=10) / fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=1)
        spectro_like_temp = fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits',
            ext=11) / fits.getdata(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=12)
        reconstructed_temp = temperature_simple_reconstruction(cluster, snap, proj, convolved=True)

    mass_weight_v_recons = np.divide((mass_weighted_temp - reconstructed_temp), mass_weighted_temp)
    theo_v_recons = np.divide((theoretical_temperature - reconstructed_temp), theoretical_temperature)
    mass_weight_v_theo = np.divide((theoretical_temperature - mass_weighted_temp), theoretical_temperature)
    mass_weight_v_spectro = np.divide((mass_weighted_temp - spectro_like_temp), mass_weighted_temp)
    theo_v_spectro = np.divide((theoretical_temperature - spectro_like_temp), theoretical_temperature)
    spectro_v_recons = np.divide((spectro_like_temp - reconstructed_temp), spectro_like_temp)

    if not convolved:
        theoretical_temperature = np.nan_to_num(theoretical_temperature)
        mass_weighted_temp = np.nan_to_num(mass_weighted_temp)
        spectro_like_temp = np.nan_to_num(spectro_like_temp)

        theoretical_temperature_contours = gaussian_filter(theoretical_temperature, sigma=3)
        mass_weight_temperature_contours = gaussian_filter(mass_weighted_temp, sigma=3)
        spectro_temperature_contours = gaussian_filter(spectro_like_temp, sigma=3)
        reconstructed_temp_contours = gaussian_filter(reconstructed_temp, sigma=3)

        mass_weight_v_recons = np.nan_to_num(mass_weight_v_recons)
        mass_weight_v_theo = np.nan_to_num(mass_weight_v_theo)
        theo_v_recons = np.nan_to_num(theo_v_recons)
        mass_weight_v_spectro = np.nan_to_num(mass_weight_v_spectro)
        theo_v_spectro = np.nan_to_num(theo_v_spectro)
        spectro_v_recons = np.nan_to_num(spectro_v_recons)
    else:
        theoretical_temperature_contours = theoretical_temperature
        mass_weight_temperature_contours = mass_weighted_temp
        spectro_temperature_contours = spectro_like_temp
        reconstructed_temp_contours = reconstructed_temp

    cluster_R500 = fits.getheader(
        f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)['R500']
    cluster_R200 = fits.getheader(
        f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)['R200']
    pix_size = fits.getheader(
        f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)['PSIZE']

    fig1 = plt.figure(figsize=(30, 30))

    if not convolved:
        vmin = 2
        vmax = 15
    else:
        vmin = 1
        vmax = 8

    ax1 = fig1.add_subplot(221, aspect='equal')
    true_map = ax1.pcolormesh(theoretical_temperature, cmap='Spectral', vmin=vmin, vmax=vmax)
    ax1.contour(theoretical_temperature_contours, levels=np.logspace(np.log10(2), np.log10(15), 10),
                colors='k')
    ax1.set_title('Theoretical temperature')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax1.add_artist(circle_R500)
    ax1.add_artist(circle_R200)
    plt.colorbar(true_map, label='T [keV]')

    ax2 = fig1.add_subplot(222, aspect='equal')
    mass_weight_map = ax2.pcolormesh(mass_weighted_temp, cmap='Spectral', vmin=vmin, vmax=vmax)
    ax2.set_title('Mass-weighted temperature')
    ax2.contour(mass_weight_temperature_contours, levels=np.logspace(np.log10(2), np.log10(15), 10), colors='k')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax2.add_artist(circle_R500)
    ax2.add_artist(circle_R200)
    plt.colorbar(mass_weight_map, label='T [keV]')

    ax3 = fig1.add_subplot(223, aspect='equal')
    spectro_like_map = ax3.pcolormesh(spectro_like_temp, cmap='Spectral', vmin=vmin, vmax=vmax)
    ax3.set_title('Spectroscopic-like temperature')
    ax3.contour(spectro_temperature_contours, levels=np.logspace(np.log10(2), np.log10(15), 10), colors='k')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax3.add_artist(circle_R500)
    ax3.add_artist(circle_R200)
    plt.colorbar(spectro_like_map, label='T [keV]')

    ax4 = fig1.add_subplot(224, aspect='equal')
    reconstructed_map = ax4.pcolormesh(reconstructed_temp, cmap='Spectral', vmin=vmin, vmax=vmax)
    ax4.set_title('Reconstructed map')
    ax4.contour(reconstructed_temp_contours, levels=np.logspace(np.log10(2), np.log10(15), 10), colors='k')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax4.add_artist(circle_R500)
    ax4.add_artist(circle_R200)
    plt.colorbar(reconstructed_map, label='T [keV]')

    chi200 = \
    fits.getheader(f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)[
        'CHI200DL']
    chi500 = \
    fits.getheader(f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)[
        'CHI500DL']
    fig1.suptitle(
        f'CL{cluster} of snap {snap}, projection {proj} with' + '$\chi_{200, DL}$' + f' = {chi200:.3f} and ' + '$\chi_{500, DL}$' + f' = {chi500:.3f}')
    if not convolved:
        plt.savefig(f'Different_temperatures_CL{cluster}_snap_{snap}_proj_{proj}.jpg')
    else:
        plt.savefig(f'Different_temperatures_convolved_CL{cluster}_snap_{snap}_proj_{proj}.jpg')

    fig2 = plt.figure(figsize=(40, 25))

    ax5 = fig2.add_subplot(231, aspect='equal')
    true_v_rec = ax5.pcolormesh(theo_v_recons, cmap='seismic', vmin=-.05, vmax=.05)
    ax5.set_title('Theoretical v Reconstructed')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='k', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='k', fill=False, lw=3)
    ax5.add_artist(circle_R500)
    ax5.add_artist(circle_R200)
    plt.colorbar(true_v_rec, label=r'$\frac{\Delta T}{T}$')

    ax6 = fig2.add_subplot(232, aspect='equal')
    mw_v_rec = ax6.pcolormesh(mass_weight_v_recons, cmap='seismic', vmin=-.05, vmax=.05)  #
    ax6.set_title('Mass weighted v Reconstructed')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='k', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='k', fill=False, lw=3)
    ax6.add_artist(circle_R500)
    ax6.add_artist(circle_R200)
    plt.colorbar(mw_v_rec, label=r'$\frac{\Delta T}{T}$')

    ax7 = fig2.add_subplot(233, aspect='equal')
    mw_v_true = ax7.pcolormesh(mass_weight_v_theo, cmap='seismic', vmin=-.05, vmax=.05)
    ax7.set_title('Mass weighted v Theoretical')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='k', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='k', fill=False, lw=3)
    ax7.add_artist(circle_R500)
    ax7.add_artist(circle_R200)
    plt.colorbar(mw_v_true, label=r'$\frac{\Delta T}{T}$')

    ax8 = fig2.add_subplot(234, aspect='equal')
    true_v_spec = ax8.pcolormesh(theo_v_spectro, cmap='seismic', vmin=-.9, vmax=.9)
    ax8.set_title('Theoretical v Spectro-like')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='k', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='k', fill=False, lw=3)
    ax8.add_artist(circle_R500)
    ax8.add_artist(circle_R200)
    plt.colorbar(true_v_spec, label=r'$\frac{\Delta T}{T}$')

    ax9 = fig2.add_subplot(235, aspect='equal')
    mw_v_spec = ax9.pcolormesh(mass_weight_v_spectro, cmap='seismic', vmin=-.9, vmax=.9)
    ax9.set_title('Mass weighted v Spectro-like')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='k', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='k', fill=False, lw=3)
    ax9.add_artist(circle_R500)
    ax9.add_artist(circle_R200)
    plt.colorbar(mw_v_spec, label=r'$\frac{\Delta T}{T}$')

    ax10 = fig2.add_subplot(236, aspect='equal')
    spec_v_recons = ax10.pcolormesh(spectro_v_recons, cmap='seismic', vmin=-.9, vmax=.9)
    ax10.set_title('Spectro-like v Reconstructed')
    circle_R500 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R500 / pix_size, color='k', fill=False, lw=3)
    circle_R200 = plt.Circle((len(theoretical_temperature) // 2, len(theoretical_temperature) // 2),
                             cluster_R200 / pix_size, color='k', fill=False, lw=3)
    ax10.add_artist(circle_R500)
    ax10.add_artist(circle_R200)
    plt.colorbar(spec_v_recons, label=r'$\frac{\Delta T}{T}$')

    fig2.suptitle(
        f'CL{cluster} of snap {snap}, projection {proj} with'+'$\chi_{200, DL}$'+f' = {chi200:.3f} and '+'$\chi_{500, DL}$'+f' = {chi500:.3f}')

    if not convolved:
        plt.savefig(f'Maps_comparison_CL{cluster}_snap_{snap}_proj_{proj}.jpg')
    else:
        plt.savefig(f'Maps_comparison_convolved_CL{cluster}_snap_{snap}_proj_{proj}.jpg')

    fig3 = plt.figure(figsize=(30, 14))
    ax11 = fig3.add_subplot(111)
    ax11.hlines(0., 0., 300., color='k', ls='--', lw=2)
    ax11.plot(mass_weight_v_theo[len(mass_weight_v_theo) // 2, len(mass_weight_v_theo) // 2:], color='crimson',
              label='$T_{mw}$')
    ax11.plot(theo_v_spectro[len(theo_v_spectro) // 2, len(theo_v_spectro) // 2:], color='darkgreen',
              label='$T_{sl}$')
    ax11.plot(theo_v_recons[len(theo_v_recons) // 2, len(theo_v_recons) // 2:], color='darkblue',
              label='$T_{recons}$')
    ax11.set_xlim(0., 260)
    ax11.set_ylabel('$\Delta T/T_{theo}$')
    ax11.text(160, -0.15,
              f'$\sigma =${np.std(mass_weight_v_theo[len(mass_weight_v_theo) // 2, len(mass_weight_v_theo) // 2:]):.3f}',
              fontsize=30)
    plt.legend()
    if not convolved:
        plt.savefig(f'Residuals_CL{cluster}_snap_{snap}_proj_{proj}.jpg')
    else:
        plt.savefig(f'Residuals_convolved_CL{cluster}_snap_{snap}_proj_{proj}.jpg')

    plt.show()


def temperature_distrib_comparison(test=False):
    files = glob(f'{nika2_resolution_maps}*proj_0.fits')

    theoretical_temps = [fits.getdata(file, ext=8)/fits.getdata(file, ext=9) for file in files]
    mass_weighted_temps = [fits.getdata(file, ext=10)/fits.getdata(file, ext=1) for file in files]
    spectro_like_temps = [fits.getdata(file, ext=11)/fits.getdata(file, ext=12) for file in files]
    # reconstructed_temps = [temperature_simple_reconstruction(file[90:94], file[100:103], 0) for file in files]
    reconstructed_temps = [temperature_simple_reconstruction(file[77:81], file[87:90], 0, convolved=True) for file in
                           files]

    masses = [fits.getheader(file, ext=1)['M500'] / 0.674 for file in files]
    pix_sizes = [fits.getheader(file, ext=1)['PSIZE'] for file in files]

    phy_R_500 = [fits.getheader(file, ext=1)['R500'] for file in files]  # convert kpc.h-1 to kpc
    pix_R_500 = phy_R_500 / np.array(pix_sizes)
    pix_R_500 = pix_R_500.astype(int)  # convert to pixel length
    center_pix = len(theoretical_temps[0]) // 2

    mean_theo_mw = []
    mean_theo_sl = []
    mean_theo_recons = []
    # std_theo_mw = []

    for i, radius in enumerate(pix_R_500):
        mask = (np.arange(0, len(theoretical_temps[i]))[np.newaxis, :] - center_pix) ** 2 + (
                np.arange(0, len(theoretical_temps[i]))[:, np.newaxis] - center_pix) ** 2 < radius
        theo_temp_in_R500 = np.mean(theoretical_temps[i][mask])
        mw_temp_in_R500 = np.mean(mass_weighted_temps[i][mask])
        sl_temp_in_R500 = np.mean(spectro_like_temps[i][mask])
        recons_temp_in_R500 = np.mean(reconstructed_temps[i][mask])

        # std_theo_temp = np.std(theoretical_temps[i][mask])
        # std_mw_temp = np.std(mass_weighted_temps[i][mask])

        mean_theo_mw.append(mw_temp_in_R500 / theo_temp_in_R500)
        mean_theo_sl.append(sl_temp_in_R500 / theo_temp_in_R500)
        mean_theo_recons.append(recons_temp_in_R500 / theo_temp_in_R500)
        # std_ratios.append(np.sqrt((std_mw_temp/theo_temp_in_R500)**2 +
        #                          (-std_theo_temp*mw_temp_in_R500/theo_temp_in_R500**2)**2))

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    # ax.errorbar(masses, mean_ratios, yerr = std_ratios, fmt = 'r^', label = '$T_{mw}$')
    ax.scatter(masses, mean_theo_mw, marker='^', color='crimson', label='$T_{mw}$')
    ax.scatter(masses, mean_theo_sl, marker='o', color='darkgreen', label='$T_{sl}$')
    ax.scatter(masses, mean_theo_recons, marker='v', color='darkblue', label='$T_{recons}$')
    ax.hlines(1., 0.1, 20., linestyle='--')
    ax.set_ylabel('$T_X/T_{theo}$')
    ax.set_xlabel(r'$M_{500} [\times 10^{14} M_\odot]$')
    ax.set_xscale('log')
    ax.set_xlim(0.5, 15)
    ax.set_ylim(0., 2.)
    plt.legend()

    plt.savefig(f'Temperature_distribution_comparison.jpg')

    plt.show()


def apply_NIKA2_resolution(test=False):
    files = glob(f'{thermo_quantity_maps}*')

    test_index = 0
    for file in files:
        test_index += 1
        cluster_name = file[103:107]  # [90:94]
        snap = file[113:116]  # [100:103]
        proj = file[122:-5]  # [109:-5]

        list_of_hdus = [fits.open(file)[0]]
        for i in range(1, 15):
            data = fits.getdata(file, ext=i)
            header = fits.getheader(file, ext=i)
            if i == 7:
                data *= (1 + 1 / 1.1568)

            nika2_resolution_kpc = (12 / 60) * Planck18.kpc_proper_per_arcmin(header['REDSHIFT']).value
            pix_size_kpc = header['PSIZE']

            nika2_resolution_pix = nika2_resolution_kpc / pix_size_kpc
            if i < 13:
                convolved_data = gaussian_filter(data, nika2_resolution_pix, mode='constant')
            else:
                convolved_data = data
            header['SPATRES'] = (nika2_resolution_pix, 'NIKA2 spatial resolution of 12", in pixel units')

            list_of_hdus.append(fits.ImageHDU(data=convolved_data, header=header))
        if test and test_index > 5:
            return 0
        final_convolved_map = fits.HDUList(list_of_hdus)
        final_convolved_map.writeto(
            f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster_name}_snap_{snap}_proj_{proj}.fits', overwrite=True)
        print(f'Wrote map for projection {proj} of CL{cluster_name} in snap {snap}')


def check_weight_maps(cluster, snap, proj):
    full_res_weight = fits.getdata(f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=12)
    full_res_Tweight = fits.getdata(f'{thermo_quantity_maps}Xray_theoretical_maps_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=11)
    full_res_spectro_like = np.nan_to_num(np.divide(full_res_Tweight, full_res_weight))

    full_res_weight_contours = gaussian_filter(full_res_weight, 3)
    full_res_Tweight_contours = gaussian_filter(full_res_Tweight, 3)
    full_res_spectro_like_contours = gaussian_filter(full_res_spectro_like, 3)

    convolved_weight = fits.getdata(f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=12)
    convolved_Tweight = fits.getdata(f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=11)
    convolved_spectro_like = np.divide(convolved_Tweight, convolved_weight)

    cluster_R500 = fits.getheader(
        f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)['R500']
    cluster_R200 = fits.getheader(
        f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)['R200']
    pix_size = fits.getheader(
        f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)['PSIZE']

    chi200 = \
    fits.getheader(f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)[
        'CHI200DL']
    chi500 = \
    fits.getheader(f'{nika2_resolution_maps}Convolved_theo_map_CL{cluster}_snap_{snap}_proj_{proj}.fits', ext=3)[
        'CHI500DL']

    fig1 = plt.figure(figsize=(40, 25))

    ax1 = fig1.add_subplot(231, aspect='equal')
    hres_weight = ax1.pcolormesh(full_res_weight, cmap='Spectral', norm=colors.LogNorm(vmin=1e8, vmax=1e16))
    ax1.set_title('Full res Weights')
    ax1.contour(full_res_weight_contours, levels=np.logspace(11, 15, 10), colors='k')
    circle_R500 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax1.add_artist(circle_R500)
    ax1.add_artist(circle_R200)
    plt.colorbar(hres_weight, label='weight')

    ax2 = fig1.add_subplot(232, aspect='equal')
    hres_Tweight = ax2.pcolormesh(full_res_Tweight, cmap='Spectral', norm=colors.LogNorm(vmin=1e8, vmax=1e16))
    ax2.set_title(r'Full res T $\times$ Weights')
    ax2.contour(full_res_Tweight_contours, levels=np.logspace(12.5, 15, 10), colors='k')
    circle_R500 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax2.add_artist(circle_R500)
    ax2.add_artist(circle_R200)
    plt.colorbar(hres_Tweight, label='T * weight')

    ax3 = fig1.add_subplot(233, aspect='equal')
    hres_spectro = ax3.pcolormesh(full_res_spectro_like, cmap='Spectral', vmin=2, vmax=15)
    ax3.set_title('Full res Spectro-like Temperature')
    ax3.contour(full_res_spectro_like_contours, levels=np.logspace(np.log10(2), np.log10(15), 10), colors='k')
    circle_R500 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax3.add_artist(circle_R500)
    ax3.add_artist(circle_R200)
    plt.colorbar(hres_spectro, label='T [keV]')

    ax4 = fig1.add_subplot(234, aspect='equal')
    conv_weight = ax4.pcolormesh(convolved_weight, cmap='Spectral', norm=colors.LogNorm(vmin=1e7, vmax=1e13))
    ax4.set_title('NIKA2 res Weights')
    ax4.contour(convolved_weight, levels=np.logspace(10, 13, 10), colors='k')
    circle_R500 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax4.add_artist(circle_R500)
    ax4.add_artist(circle_R200)
    plt.colorbar(conv_weight, label='weight')

    ax5 = fig1.add_subplot(235, aspect='equal')
    conv_Tweight = ax5.pcolormesh(convolved_Tweight, cmap='Spectral', norm=colors.LogNorm(vmin=1e8, vmax=1e13))
    ax5.set_title(r'NIKA2 res T $\times$ Weights')
    ax5.contour(convolved_Tweight, levels=np.logspace(11, 14, 10), colors='k')
    circle_R500 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax5.add_artist(circle_R500)
    ax5.add_artist(circle_R200)
    plt.colorbar(conv_Tweight, label='T * weight')

    ax6 = fig1.add_subplot(236, aspect='equal')
    conv_spectro = ax6.pcolormesh(convolved_spectro_like, cmap='Spectral', vmin=1, vmax=8)
    ax6.set_title('NIKA2 res Spectro-like Temperature')
    ax6.contour(convolved_spectro_like, levels=np.logspace(np.log10(2), np.log10(15), 10), colors='k')
    circle_R500 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R500 / pix_size, color='w', fill=False, lw=3)
    circle_R200 = plt.Circle((len(full_res_weight) // 2, len(full_res_weight) // 2),
                             cluster_R200 / pix_size, color='w', fill=False, lw=3)
    ax6.add_artist(circle_R500)
    ax6.add_artist(circle_R200)
    plt.colorbar(conv_spectro, label='T [keV]')

    fig1.suptitle(
        f'CL{cluster} of snap {snap}, projection {proj} with'+'$\chi_{200, DL}$'+f' = {chi200:.3f} and '+'$\chi_{500, DL}$'+f' = {chi500:.3f}')

    plt.savefig(f'Weight_maps_CL{cluster}_snap_{snap}_proj_{proj}.jpg')
    plt.show()