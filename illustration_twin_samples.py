import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pandas as pd
from glob import glob
from matplotlib import colors


members_in_TS = pd.read_csv('data/NIKA2_Twin_Samples_selection.txt', sep = "  ", header=None, dtype=str)
members_in_TS = members_in_TS.drop(index=[18,19,31])

clusters_IDs_in_TS1 = members_in_TS.loc[:,0].values
clusters_snaps_in_TS1 = members_in_TS.loc[:,2].values

IDs_of_TS1 = [[clusters_IDs_in_TS1[i],clusters_snaps_in_TS1[i]] for i in range(len(clusters_IDs_in_TS1))]

data_filenames_TS1 = [f'/cluster/data/NIKA2/Twin_Sample/NIKA2_mock_maps/NIKA2_like_maps_wo_noise/snap_{ID[1][1:]}/CL0{ID[0]}_0-0-0.fits' for ID in IDs_of_TS1]

print(len(data_filenames_TS1))

fig = plt.figure(figsize=(20,13))
for i, data_filename in enumerate(data_filenames_TS1):
    data = fits.getdata(data_filename)
    ax = fig.add_subplot(5,9,i+1)
    ax.imshow(data)#, norm=colors.LogNorm())
    ax.contour(data, levels=np.logspace(-6,-3,15), colors='k')

plt.tight_layout()
plt.savefig('data/illustration_TS1.jpg', dpi=300)
plt.show()
