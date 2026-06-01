'''
Calculate inverse concentration index for NSA galaxies
'''

################################################################################
# IMPORTS
################################################################################

from astropy.table import Table
from astropy.io import fits

import numpy as np
import numpy.ma as ma

from color_gradient_functions import r_pct_SGA_cog, calc_cinv


################################################################################
# data
################################################################################


GV_FOLDER = '/pscratch/sd/n/nravi/GV_classification/'

# nsa table
NSA_FN = GV_FOLDER + '/nsa_v1_0_1_cd_v4.fits'

SAVE_FN = GV_FOLDER + 'nsa_v1_0_1_cd_v4_cinv.fits'


NSA = Table.read(NSA_FN)

################################################################################
# add columns
################################################################################

NSA['R50_i'] = np.ones(len(NSA))*np.nan
NSA['R90_i'] = np.ones(len(NSA))*np.nan
NSA['cinv'] = np.ones(len(NSA))*np.nan




for i in range(len(NSA)):

    R50_i = np.nan
    R90_i = np.nan
    cinv = np.nan

    i_mtot = NSA['i_mtot'][i]
    i_m0 = NSA['i_m0'][i]
    i_a1 = NSA['i_a1'][i]
    i_a2 = NSA['i_a2'][i]
        
    if not ma.is_masked(NSA['i_mtot'][i]):

        R50_i = r_pct_SGA_cog(50, i_mtot, i_m0, i_a1, i_a2)
        R90_i = r_pct_SGA_cog(90, i_mtot, i_m0, i_a1, i_a2)

        cinv = R50_i/R90_i

    else:
        continue
    
    ################################################################################
    # add to table
    ################################################################################

    NSA['R50_i'][i] = R50_i
    NSA['R90_i'][i] = R90_i
    NSA['cinv'][i] = cinv



NSA.write(SAVE_FN, format='fits', overwrite=True)