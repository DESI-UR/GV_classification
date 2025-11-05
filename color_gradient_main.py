'''

Calculate color gradient for NSA galaxies

'''

################################################################################
# IMPORTS
################################################################################

from astropy.table import Table
from astropy.io import fits

from scipy.optimize import curve_fit, minimize

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from color_gradient_functions import extract_cog_data, calculate_color_gradient

################################################################################
# data
################################################################################


DATA_FOLDER = '/Users/nityaravi/Documents/Research/RotationCurves/data/'

# folder with curve of growth files
COG_FOLDER = DATA_FOLDER + 'nsa/cog/'

# location to save plots
PLOT_DIR = '/Users/nityaravi/Documents/GitHub/GV_classification/'

# location to save covariance matrices
COV_FOLDER = '/Users/nityaravi/Documents/GitHub/GV_classification/'

# nsa table
NSA_FN = DATA_FOLDER + 'nsa_v1_0_1.fits'

# new table save loc
COG_SAVE_FN = 'test_cog.fits'

# read in nsa table
NSA = Table.read(NSA_FN, format='fits')

NSA = NSA[37630:37632]

# add columns to table
new_cols = ['g_mtot', 'g_m0', 'g_a1', 'g_a2', 
            'i_mtot', 'i_m0', 'i_a1', 'i_a2',
            'NSA_cd']


for col in new_cols:

    NSA[col] = np.ones(len(NSA))*np.nan


################################################################################


################################################################################
# color gradient calculation
################################################################################

count = 0

for i in range(len(NSA)):

    if count % 1000 == 0:
        print(count)

    count += 1

    iauname = NSA['IAUNAME'][i]
    Rpet = NSA['ELPETRO_THETA'][i]

    ############################################################################
    # get curve of growth files
    # --------------------------------------------------------------------------

    i_r, i_f = extract_cog_data(iauname, 'i', COG_FOLDER)
    g_r, g_f = extract_cog_data(iauname, 'g', COG_FOLDER)

    if i_r is None or g_r is None:

        continue

    ############################################################################
    # get cog fit params and color gradient
    # --------------------------------------------------------------------------

    try:

        i_mag_fit, i_mag_cov, g_mag_fit, g_mag_cov, cd = calculate_color_gradient(i_f, 
                                                                              g_f,
                                                                              i_r,
                                                                              g_r,
                                                                              Rpet,
                                                                              iauname,
                                                                              PLOT_DIR)
        
    except:
        print('Fitting ', iauname, ' failed!')
        continue


    ############################################################################
    # save cov matrices
    # --------------------------------------------------------------------------

    np.save(COV_FOLDER + '/' + iauname + '_i_cov.npy', i_mag_cov)
    np.save(COV_FOLDER + '/' + iauname + '_g_cov.npy', g_mag_cov)


    ############################################################################
    # write cog paras and color gradient to table
    # --------------------------------------------------------------------------

    NSA['i_mtot'][i] = i_mag_fit[0]
    NSA['i_m0'][i] = i_mag_fit[1]
    NSA['i_a1'][i] = i_mag_fit[2]
    NSA['i_a2'][i] = i_mag_fit[3]

    NSA['g_mtot'][i] = g_mag_fit[0]
    NSA['g_m0'][i] = g_mag_fit[1]
    NSA['g_a1'][i] = g_mag_fit[2]
    NSA['g_a2'][i] = g_mag_fit[3]

    NSA['NSA_cd'][i] = cd

################################################################################


################################################################################
# Save table
# --------------------------------------------------------------------------

NSA.write(COG_SAVE_FN, format='fits', overwrite=True)