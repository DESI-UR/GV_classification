'''
Plot the color histogram of a set of galaxies.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import numpy as np

import matplotlib.pyplot as plt
################################################################################




################################################################################
# Import data
#-------------------------------------------------------------------------------
#galaxies_fileName = '../../data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt'
#galaxies_fileName = '../../data/NSA_v1_0_1_VAGC_CMDJan2020_vflag-V2-VF.fits'
galaxies_fileName = '../../data/NSA_v0_1_2_VAGC.fits'

#galaxies_table = Table.read(galaxies_fileName, format='ascii.commented_header')
galaxies_table = Table.read(galaxies_fileName, format='fits')
################################################################################




################################################################################
# Plotting
#-------------------------------------------------------------------------------
# Formatting
tSize = 18 # text size
lWidth = 4 # line width

# Plot parameters
cmin = 0
cmax = 4
cstep = 0.01


plt.figure()

plt.hist(#galaxies_table['u_r_NSA'], 
         galaxies_table['u_r'], 
         bins=np.arange(cmin,cmax,cstep), 
         color='0.25', 
         edgecolor='none')

plt.xlim((cmin, cmax))

plt.xlabel('u - r', fontsize=tSize)
plt.ylabel('Number of galaxies', fontsize=tSize)

ax = plt.gca()
ax.tick_params(labelsize=tSize, length=10., width=3.)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

#plt.show()
plt.savefig('Figures/Histograms/ur_hist-NSAv012_kcorr.eps', 
            format='eps', 
            dpi=300)
################################################################################