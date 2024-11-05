'''
Plot the color vs. color gradient, and the color vs. inverse concentration 
index, for a set of galaxies.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
################################################################################





################################################################################
# Galaxy data
#-------------------------------------------------------------------------------
galaxies_fileName = '/Users/kellydouglass/Documents/Research/data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_HI100.txt'

galaxies_table = Table.read(galaxies_fileName, format='ascii.commented_header')
################################################################################





################################################################################
# Bin galaxies for contours
#-------------------------------------------------------------------------------
ur_bins = np.linspace(0, 4, 100)

# Color - color gradient
A_N, Cedges, Gedges = np.histogram2d(galaxies_table['u_r'], 
                                     galaxies_table['cd'], 
                                     bins=(ur_bins, np.arange(-0.8, 0.8, 0.1)))
Cmesh, Gmesh = np.meshgrid(Cedges[:-1], Gedges[:-1], indexing='ij')

# Color - inverse concentration index
B_N, Cedges2, Iedges = np.histogram2d(galaxies_table['u_r'], 
                                      galaxies_table['conx1'], 
                                      bins=(ur_bins, np.arange(0, 1, 0.1)))
Cmesh2, Imesh = np.meshgrid(Cedges2[:-1], Iedges[:-1], indexing='ij')
################################################################################





################################################################################
# Plot
#-------------------------------------------------------------------------------
# Plot formatting
tSize = 14 # text size
lWidth = 4 # line width


#-------------------------------------------------------------------------------
# Color - color gradient
#-------------------------------------------------------------------------------
plt.figure(1)

plt.contourf(Cmesh, Gmesh, A_N, 50, cmap=cm.get_cmap('bone_r'))

plt.xlabel('u - r', fontsize=tSize)
plt.ylabel('$\Delta$ (g - i)', fontsize=tSize)

plt.axis([1, 3.5, -0.6, 0.2])

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
plt.savefig('Figures/color-color_grad/ur_v_dgi.eps', format='eps', dpi=500)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Color - inverse concentration index
#-------------------------------------------------------------------------------
plt.figure(2)

plt.contourf(Cmesh2, Imesh, B_N, 50, cmap=cm.get_cmap('bone_r'))

plt.xlabel('u - r', fontsize=tSize)
plt.ylabel('$c_{inv}$', fontsize=tSize)

plt.axis([1, 3.5, 0.1, 0.6])

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
plt.savefig('Figures/color-color_grad/ur_v_cinv.eps', format='eps', dpi=500)
################################################################################
