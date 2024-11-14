'''
Calculate galaxy morphology type (aimc)

Based off the values published in the KIAS-VAGC
'''

import numpy as np


################################################################################
#
#   CALCULATE AIMC
#
################################################################################


def SDSS_aimc(u_r, Dg_i):

    # Formula derived by Greg Zengilowski
    aimc = np.arctan2(-Dg_i + 0.260, u_r - 1.343)

    # Convert from radians to degrees
    aimc = aimc*180/np.pi

    # Shift to 0-360
    aimc[aimc < 0] += 360

    return aimc




def mySDSS_aimc(u_r, Dg_i):

    ur_shift = -2.067

    aimc = np.arctan2(-Dg_i, u_r + ur_shift)

    # Convert from radians to degrees
    aimc = aimc*180/np.pi

    # Shift to 0-360
    aimc[-Dg_i <= 0] += 360

    '''
    aimc = np.arctan(-Dg_i/(u_r + ur_shift))

    # Convert from radians to degrees
    aimc = -aimc*180/np.pi

    for i in xrange(len(u_r)):
        if u_r[i] + ur_shift < 0:
            aimc[i] = aimc[i] + 180
        else:
            if -Dg_i[i] < 0:
                aimc[i] = aimc[i] + 360
    '''

    return aimc



def my_aimc(grad, color):
    
    grad_shift = 0.3
    color_shift = -1
    
    aimc = np.arctan2(-grad + grad_shift, color + color_shift)
    
    # Need to convert from radians to degrees
    aimc = aimc*180/np.pi
    
    # Need to map [-180,0) to [180,360)
    aimc[aimc < 0] += 360
    
    return aimc



def my_aimc_vertical(grad, color):
    
    grad_shift = 0.3
    color_shift = -1
    
    aimc = np.arctan2(-grad + grad_shift, color + color_shift) + 0.5*np.pi
    
    # Need to convert from radians to degrees
    aimc = aimc*180/np.pi
    
    # Need to map [-180,0) to [180,360)
    aimc[aimc < 0] += 360
    
    return aimc



def my_aimc_median(grad, color):

    # Median values in the SDSS DR7 galaxy sample via the KIAS-VAGC (Choi et al. 
    # 2010), using only those galaxies with good photometry (u-r > -35, color 
    # gradient != -9.99, -10)
    grad_shift = -0.112
    color_shift = 2.484

    aimc = np.arctan2(-grad + grad_shift, color - color_shift) + 0.5*np.pi
    
    # Need to convert from radians to degrees
    aimc = aimc*180/np.pi
    
    # Need to map [-180,0) to [180,360)
    aimc[aimc < 0] += 360
    
    return aimc





if __name__ == '__main__':

    from astropy.table import Table

    ############################################################################
    #   IMPORT DATA
    ############################################################################


    galaxy_filename = '../../Data/dwarf_voidstatus_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_ALFALFA_HI70.txt'
    #galaxy_filename = '../../Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag.txt'

    galaxies = Table.read(galaxy_filename, format='ascii.commented_header')

    galaxies['my_aimc'] = my_aimc(galaxies['u_r'], galaxies['cd'])


    ############################################################################
    #   SAVE DATA
    ############################################################################


    galaxies.write(galaxy_filename[:-4]+'_aimc.txt', format='ascii.commented_header')