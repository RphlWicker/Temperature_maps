import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib import colors
from glob import glob

test_maps = glob('/cluster/data/NIKA2/Twin_Sample/NIKA2_mock_maps/NIKA2_like_maps_wo_noise/snap_101/CL0027_*.fits')

fig = plt.figure(figsize=(30,30))
for i, b_map in enumerate(test_maps[:16]):
    data = fits.getdata(b_map)
    ax = fig.add_subplot(4,4,i+1)
    ax.imshow(data, norm=colors.LogNorm())
    ax.contour(data, levels=np.logspace(-6,-3,15), colors='k')
    ax.set_title(b_map[82:-5])  

plt.subplots_adjust(top=0.978, bottom=0.028, left=0., right=1., hspace=0.221, wspace=0.)
plt.show()
