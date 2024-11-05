'''
Plot the color vs. color gradient for a set of galaxies, colored by their aimc
values.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.cm as cm
################################################################################





################################################################################
# Galaxy data
#-------------------------------------------------------------------------------
#galaxies_fileName = '../../Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt'
galaxies_fileName = '/Users/kellydouglass/Documents/Research/data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_NSAv012_CMDJan2020.txt'
#galaxies_fileName = '/Users/kellydouglass/Documents/Research/data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_HI100.txt'

galaxies_table = Table.read(galaxies_fileName, format='ascii.commented_header')
################################################################################





################################################################################
# Calculate the galaxies' aimc values
#-------------------------------------------------------------------------------
from aimc_SDSS import my_aimc, my_aimc_vertical

galaxies_table['my_aimc'] = my_aimc(galaxies_table['cd'], 
                                             galaxies_table['u_r'])
################################################################################





################################################################################
# Plot
#
# Note: The plotted data is downsampled to reduce the number of points in the 
#       plot.  (With all galaxies, the figure size is almost 72 MB.)
#-------------------------------------------------------------------------------
# Plot formatting
tSize = 14 # text size
lWidth = 4 # line width


plt.figure()

plt.scatter(galaxies_table['u_r'][::10], 
            galaxies_table['cd'][::10], 
            s=1, 
            c=galaxies_table['aimc'][::10], 
            #alpha=0.5,
            edgecolors='none')
plt.plot([1,1,2.6,4], [0.8,0.3,-0.15,-0.15], 'k')

cbar = plt.colorbar()
cbar.ax.set_ylabel('aimc', fontsize=tSize)
cbar.ax.tick_params(labelsize=tSize, width=3)

plt.xlabel('u - r', fontsize=tSize)
plt.ylabel('$\Delta$ (g - i)', fontsize=tSize)

plt.axis([0,4,-0.8,0.8])

ax = plt.gca()
ax.tick_params(labelsize=tSize, length=10, width=3)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

#plt.show()
plt.savefig('Figures/color-color_grad/ur_v_dgi_aimcKIAS.eps', format='eps', dpi=500)
################################################################################
