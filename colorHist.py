'''
Plot the color histogram of a set of galaxies, splitting them into their CMD 
categories.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table

import numpy as np

import matplotlib.pyplot as plt

from CMD_classification import CMD_class, CMD_class_Jan2020
################################################################################




################################################################################
# Galaxy data
#-------------------------------------------------------------------------------
#galaxies_fileName = '../../Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt'
galaxies_fileName = '../../data/NSA_v1_0_1_VAGC_CMDJan2020_vflag-V2-VF.fits'

#galaxies_table = Table.read(galaxies_fileName, format='ascii.commented_header')
galaxies_table = Table.read(galaxies_fileName, format='fits')
################################################################################




################################################################################
# Set color to be plotted
#-------------------------------------------------------------------------------
color_data = galaxies_table['NUV_r']
################################################################################




################################################################################
# Separate galaxies by their CMD classification
#-------------------------------------------------------------------------------
'''
# Thesis classification
#-------------------------------------------------------------------------------
# Green valley
gboolarray = np.logical_and.reduce((galaxies_table['aimc'] < 25, 
                                    galaxies_table['aimc'] != 1, 
                                    galaxies_table['aimc'] != 2))
# Red sequence
bboolarray = galaxies_table['aimc'] > 25

# Blue cloud
rboolarray = np.logical_or(galaxies_table['aimc'] == 1, 
                           galaxies_table['aimc'] == 2)
#-------------------------------------------------------------------------------
'''

#-------------------------------------------------------------------------------
# January 2020 classification
#-------------------------------------------------------------------------------
galaxies_table['CMD_class'] = CMD_class_Jan2020(galaxies_table['u_r_KIAS'], 
                                                galaxies_table['cd'], 
                                                galaxies_table['conx1'])
#-------------------------------------------------------------------------------

'''
#-------------------------------------------------------------------------------
# June 2020 classification
#-------------------------------------------------------------------------------
galaxies_table['CMD_class'] = CMD_class(galaxies_table['u_r_KIAS'], 
                                        galaxies_table['cd'], 
                                        galaxies_table['conx1'], 
                                        galaxies_table['prmag_KIAS'])
#-------------------------------------------------------------------------------
'''

# Green valley
gboolarray = galaxies_table['CMD_class'] == 2
# Blue cloud
bboolarray = galaxies_table['CMD_class'] == 1
# Red sequence
rboolarray = galaxies_table['CMD_class'] == 3

GVdata = color_data[gboolarray]
Bdata = color_data[bboolarray]
Rdata = color_data[rboolarray]
################################################################################




################################################################################
# Plotting
#-------------------------------------------------------------------------------
# Formatting
tSize = 18 # text size
lWidth = 4 # line width

# Plot parameters
cmin = 0
cmax = 10
cstep = 0.04 # 0.02

cbins = np.arange(cmin, cmax, cstep)


plt.figure()

plt.hist(color_data, 
         bins=cbins, 
         color='0.5', 
         edgecolor='none', 
         label='SDSS DR7')
plt.hist(Rdata, 
         bins=cbins, 
         color='r', 
         edgecolor='none', 
         alpha=0.5, label='Red sequence')
plt.hist(Bdata, 
         bins=cbins, 
         color='b', 
         edgecolor='none',
         alpha=0.5, 
         label='Blue cloud')
plt.hist(GVdata, 
         bins=cbins, 
         color='g', 
         edgecolor='none',
         alpha=0.75, 
         label='Green valley')

plt.xlim((cmin, cmax))

plt.xlabel('NUV - r', fontsize=tSize)
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

plt.show()
#plt.savefig('Histograms/NUVr_myCMD_hist.eps', format='eps', dpi=500)
################################################################################