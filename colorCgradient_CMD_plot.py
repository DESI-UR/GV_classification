'''
Plot the color vs. color gradient for a set of galaxies, separated by their CMD 
classification.
'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from astropy.table import Table
import astropy.constants as const

import numpy as np

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/spirals/')
from dark_matter_mass_v1 import rot_fit_BB

# from CMD_classification import CMD_class, CMD_class_Jan2020

import matplotlib.pyplot as plt
import matplotlib.cm as cm
################################################################################




################################################################################
# Constants
#-------------------------------------------------------------------------------
H0 = 100
################################################################################





################################################################################
# Galaxy data
#-------------------------------------------------------------------------------
#galaxies_fileName = '../../Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt'
#galaxies_fileName = '/Users/kellydouglass/Documents/Research/data/kias1033_5_MPAJHU_ZdustOS_NSAv012_CMDJan2020.txt'
#galaxies_fileName = '/Users/kellydouglass/Documents/Research/data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_HI100.txt'
#galaxies_fileName = '/Users/kellydouglass/Desktop/Pipe3D-master_file_vflag_10_smooth2p27_N2O2_noWords.txt'
#galaxies_fileName = '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/spirals/Pipe3D-master_file_vflag_BB_minimize_chi10_smooth2p27_mapFit_N2O2_HIdr2_noWords_v5.txt'
#galaxies_fileName = '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/spirals/DRP-master_file_vflag_BB_smooth1p85_mapFit_N2O2_HIdr2_morph_noWords_v6.txt'
#galaxies_fileName = '../../data/NSA_v1_0_1_VAGC.fits'
galaxies_fileName = '/Users/kellydouglass/Documents/Research/Rotation_curves/Nitya_Ravi/master_table_H_alpha_BB_HI_H2_MxCG_R90_CMD_ZPG16R_SFR_MZ.txt'

data = Table.read(galaxies_fileName, format='ascii.commented_header')
#galaxies_table = Table.read(galaxies_fileName, format='fits')
################################################################################
print('Total # of galaxies:', len(data))




################################################################################
# Calculate velocity at R90
#-------------------------------------------------------------------------------
dist_to_galaxy_Mpc = (const.c.to('km/s')*data['NSA_redshift']/H0).value
dist_to_galaxy_kpc = dist_to_galaxy_Mpc*1000

data['R90_kpc'] = dist_to_galaxy_kpc*np.tan(data['NSA_elpetro_th90']*(1./60)*(1./60)*(np.pi/180))

data['V90_kms'] = rot_fit_BB(data['R90_kpc'], 
                             [data['Vmax_map'], 
                              data['Rturn_map'], 
                              data['alpha_map']])
################################################################################



'''
################################################################################
# Calculate mass ratio
#-------------------------------------------------------------------------------
data['M90_Mdisk_ratio'] = 10**(data['M90_map'] - data['M90_disk_map'])
################################################################################
'''


'''
################################################################################
# Use only those galaxies with valid rotation curves
#
# Note: This is for the second rotation curve paper and should not be used for 
# all instances of generating this plot.
#-------------------------------------------------------------------------------
#bad_boolean = data['curve_used'] == -99
bad_boolean = np.logical_or.reduce([data['M90_map'] == -99, 
                                    data['M90_disk_map'] == -99, 
                                    data['alpha_map'] > 99, 
                                    data['ba_map'] > 0.998, 
                                    data['V90_kms']/data['Vmax_map'] < 0.9, 
                                    (data['Tidal'] & (data['DL_merge'] > 0.97)), 
                                    data['map_frac_unmasked'] < 0.05, 
                                    (data['map_frac_unmasked'] > 0.13) & (data['DRP_map_smoothness'] > 1.96), 
                                    (data['map_frac_unmasked'] > 0.07) & (data['DRP_map_smoothness'] > 2.9), 
                                    (data['map_frac_unmasked'] > -0.0638*data['DRP_map_smoothness'] + 0.255) & (data['DRP_map_smoothness'] > 1.96), 
                                    data['M90_Mdisk_ratio'] > 1050])

galaxies_table = data[~bad_boolean]
################################################################################
'''



################################################################################
# Use only those with valid rotation curves
#
# Note: This is for Nitya's first paper and should not be used for all instances 
# of generating this plot.
#-------------------------------------------------------------------------------
bad_boolean = np.logical_or.reduce([data['Vmax_map'] == -999, 
                                    data['M90_disk_map'] == -999, 
                                    data['V90_kms'] < 10, 
                                    data['V90_kms'] > 1000, 
                                    data['alpha_map'] > 99, 
                                    data['Vmax_err_map']/data['Vmax_map'] > 2])

galaxies_table = data[~bad_boolean]
################################################################################
print('Galaxies w/ good fits:', np.sum((data['M90_map'] > 0) & (data['M90_disk_map'] > 0) & (data['Vmax_map'] > 0)))
print('# galaxies after M90 cut:', np.sum((data['M90_map'] > 0) & (data['M90_disk_map'] > 0) & (data['Vmax_map'] > 0) & (data['M90_map'] >= 9)))
print('# galaxies after alpha cut:', np.sum((data['M90_map'] > 0) & (data['M90_disk_map'] > 0) & (data['Vmax_map'] > 0) & (data['M90_map'] >= 9) & (data['alpha_map'] <= 99)))
print('# galaxies after Vmax_err cut:', np.sum((data['M90_map'] > 0) & (data['M90_disk_map'] > 0) & (data['Vmax_map'] > 0) & (data['M90_map'] >= 9) & (data['alpha_map'] <= 99) & (data['Vmax_err_map']/data['Vmax_map'] <= 2)))
print(len(galaxies_table))
exit()




################################################################################
# Separate galaxies by aimc
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

'''
#-------------------------------------------------------------------------------
# January 2020 classification
#
# THIS IS THE ONE THAT WE'VE BEEN USING!
#-------------------------------------------------------------------------------
galaxies_table['CMD_class'] = CMD_class_Jan2020(galaxies_table['u_r_KIAS'], 
                                                galaxies_table['cd'], 
                                                galaxies_table['conx1'])
#-------------------------------------------------------------------------------
'''
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


GVdata = galaxies_table[gboolarray]
Bdata = galaxies_table[bboolarray]
Rdata = galaxies_table[rboolarray]
################################################################################





################################################################################
# Plot
#
# Note: The plotted data is downsampled to reduce the number of points in the 
#       plot.  (With all galaxies, the figure size is almost 72 MB.)
#-------------------------------------------------------------------------------
# Plot formatting
tSize = 14 # text size
lWidth = 3 # line width
mSize = 3  # marker size


plt.figure(tight_layout=True)

plt.plot(Rdata['u_r_KIAS'],#[::10], 
         Rdata['cd'],#[::10], 
         'ro', # 'r^'
         fillstyle='none', 
         markersize=mSize,
         #alpha=0.1,
         label='Red sequence')

plt.plot(Bdata['u_r_KIAS'],#[::10], 
         Bdata['cd'],#[::10], 
         'b+', # 'b.'
         markersize=mSize+3, 
         #alpha=0.1, 
         label='Blue cloud')

plt.plot(GVdata['u_r_KIAS'],#[::10], 
         GVdata['cd'],#[::10], 
         'g*', 
         markersize=mSize,
         #alpha=0.1,
         label='Green valley')

# Bounding area defined by Park05
plt.plot([1,1,2.6,4], [0.8,0.3,-0.15,-0.15], 'k')

# plt.xlabel('u - r (KIAS)', fontsize=tSize)
plt.xlabel('u - r', fontsize=tSize)
plt.ylabel('$\Delta$ (g - i)', fontsize=tSize)

plt.legend()#fontsize=tSize)

#plt.axis([0,4,-0.8,0.8])
plt.axis([0, 4, -0.6, 0.5])

ax = plt.gca()
ax.tick_params(labelsize=tSize)#, length=10, width=lWidth)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_linewidth(lWidth)
# ax.spines['left'].set_linewidth(lWidth)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# plt.show()

plt.savefig(#'Figures/color-color_grad/ur_v_dgi_JanCMD_MaNGA_paper2redo_v6.eps', 
            #'Figures/color-color_grad/ur_v_dgi_thesisCMD.eps',
            #'Figures/color-color_grad/ur_v_dgi_JanCMD.eps', 
            'Figures/color-color_grad/ur_v_dgi_JanCMD_MaNGA_NityaPaper1_v1.eps', 
            format='eps', 
            dpi=300)

################################################################################
