


import numpy as np

from aimc_SDSS import my_aimc, my_aimc_vertical, my_aimc_median




def CMD_class_median(color, grad, cinv, rmag):
    '''
    Determine if a galaxy is in the blue cloud (BC), green valley (GV), or red 
    sequence (RS).

    For use with SDSS photometry; requires:
      - color: u-r
      - color gradient: Delta g-i
      - inverse concentration index


    PARAMETERS
    ==========

    color : list or array of length N
        u-r color of the galaxies

    grad : list or array of length N
        g-i color gradient of the galaxies

    cinv : list or array of length N
        inverse concentration indices of the galaxies

    rmag : list or array of length N
        Apparent Petrosian magnitude of the galaxies


    RETURNS
    =======

    CMDclass : numpy array of length N
        CMD classification for each galaxy
          - 3 = Red sequence
          - 2 = Green vallye
          - 1 = Blue cloud
    '''


    ############################################################################
    # Initialize output array
    #---------------------------------------------------------------------------
    CMDclass = -1*np.ones(len(color), dtype=int)
    ############################################################################



    ############################################################################
    # Calculate the aimc values
    #---------------------------------------------------------------------------
    aimc = my_aimc_median(grad, color)
    ############################################################################



    ############################################################################
    # Magnitude bins
    #
    # According to Park05, the morphological classification for the galaxies is 
    # slightly different, depending on their apparent magnitudes.
    #---------------------------------------------------------------------------
    boolean_14p5_16 = rmag < 16
    boolean_16_16p5 = np.logical_and(rmag >= 16, rmag < 16.5)
    boolean_16p5_17 = np.logical_and(rmag >= 16.5, rmag < 17)
    boolean_17_17p5 = rmag >= 17
    ############################################################################



    ############################################################################
    # Define the inverse concentration index limit on the intersection of the 
    # skew-normal mixture model
    #
    # Park05 adds a supplemental condition on early-type galaxies based on their 
    # inverse concentration indices.  They used a flat value for each of the 
    # four magnitude bins.  Here, we fit the distribution of the early galaxies 
    # in u-r with a skew-normal mixture model, and define the maximum inverse 
    # concentration index for the early galaxies (the eventual red sequence) 
    # based on the intersection of these two populations.
    #
    # The skew-normal mixture model parameters and intersection points were 
    # found in the morph_classification-2.ipynb Jupyter notebook.
    #---------------------------------------------------------------------------
    cinv_bright = 0.36389885134100214
    cinv_16_16p5 = 0.37696747603194236
    cinv_16p5_17 = 0.3839725657693284
    cinv_faint = 0.3858342667856255
    ############################################################################



    ############################################################################
    # Morphological classification
    #
    # Based on the morphological classification of Park & Choi (2005)
    #---------------------------------------------------------------------------
    # Early type
    #---------------------------------------------------------------------------
    early_boolean1 = np.logical_and(color > 1, grad > 0.3)

    early_boolean2_14p5_16 = np.logical_and.reduce((color > 1, 
                                                    color <= 2.6, 
                                                    grad > -0.28125*color + 0.58125))
    early_boolean2_16_17 = np.logical_and.reduce((color > 1,
                                                  color <= 2.65,
                                                  grad > -0.290909*color + 0.590909))
    early_boolean2_17_17p5 = np.logical_and.reduce((color > 1,
                                                    color <= 2.7,
                                                    grad > -0.28236*color + 0.582353))

    early_boolean3_14p5_16 = np.logical_and(color > 2.60, grad > -0.15)
    early_boolean3_16_16p5 = np.logical_and(color > 2.65, grad > -0.15)
    early_boolean3_16p5_17 = np.logical_and(color > 2.65, grad > -0.25)
    early_boolean3_17_17p5 = np.logical_and(color > 2.70, grad > -0.35)

    early_boolean_14p5_16_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_14p5_16, 
                                                       early_boolean3_14p5_16))
    early_boolean_16_16p5_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_16_17, 
                                                       early_boolean3_16_16p5))
    early_boolean_16p5_17_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_16_17, 
                                                       early_boolean3_16p5_17))
    early_boolean_17_17p5_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_17_17p5, 
                                                       early_boolean3_17_17p5))

    # Require early-type galaxies to have inverse concentration indices smaller 
    # than some value
    early_boolean_14p5_16 = np.logical_and(early_boolean_14p5_16_area, 
                                           cinv < cinv_bright)
    early_boolean_16_16p5 = np.logical_and(early_boolean_16_16p5_area, 
                                           cinv < cinv_16_16p5)
    early_boolean_16p5_17 = np.logical_and(early_boolean_16p5_17_area, 
                                           cinv < cinv_16p5_17)
    early_boolean_17_17p5 = np.logical_and(early_boolean_17_17p5_area, 
                                           cinv < cinv_faint)

    # Galaxies that would normally have become late-types because of the above 
    # restriction on cinv (conx1) are instead being classified as GV galaxies
    gv_boolean_14p5_16 = np.logical_and(early_boolean_14p5_16_area, 
                                        cinv >= cinv_bright)
    gv_boolean_16_16p5 = np.logical_and(early_boolean_16_16p5_area, 
                                        cinv >= cinv_16_16p5)
    gv_boolean_16p5_17 = np.logical_and(early_boolean_16p5_17_area, 
                                        cinv >= cinv_16p5_17)
    gv_boolean_17_17p5 = np.logical_and(early_boolean_17_17p5_area, 
                                        cinv >= cinv_faint)

    # All early-type galaxies (what will become the red sequence)
    early_galaxies_boolean = np.logical_or.reduce((np.logical_and(boolean_14p5_16, 
                                                                  early_boolean_14p5_16), 
                                                   np.logical_and(boolean_16_16p5, 
                                                                  early_boolean_16_16p5), 
                                                   np.logical_and(boolean_16p5_17, 
                                                                  early_boolean_16p5_17), 
                                                   np.logical_and(boolean_17_17p5, 
                                                                  early_boolean_17_17p5)))
    #---------------------------------------------------------------------------
    # Late type
    #---------------------------------------------------------------------------
    late_boolean_14p5_16 = np.logical_not(early_boolean_14p5_16)
    late_boolean_16_16p5 = np.logical_not(early_boolean_16_16p5)
    late_boolean_16p5_17 = np.logical_not(early_boolean_16p5_17)
    late_boolean_17_17p5 = np.logical_not(early_boolean_17_17p5)
    #---------------------------------------------------------------------------
    late_galaxies_boolean = np.logical_or.reduce((np.logical_and(boolean_14p5_16, 
                                                                 late_boolean_14p5_16),
                                                  np.logical_and(boolean_16_16p5, 
                                                                 late_boolean_16_16p5),
                                                  np.logical_and(boolean_16p5_17, 
                                                                 late_boolean_16p5_17),
                                                  np.logical_and(boolean_17_17p5, 
                                                                 late_boolean_17_17p5)))
    #---------------------------------------------------------------------------
    # Early-type galaxies in the Green Valley
    #---------------------------------------------------------------------------
    gv_early_boolean = np.logical_or.reduce((np.logical_and(boolean_14p5_16, 
                                                            gv_boolean_14p5_16), 
                                             np.logical_and(boolean_16_16p5, 
                                                            gv_boolean_16_16p5), 
                                             np.logical_and(boolean_16p5_17, 
                                                            gv_boolean_16p5_17), 
                                             np.logical_and(boolean_17_17p5, 
                                                            gv_boolean_17_17p5)))
    ############################################################################



    ############################################################################
    # Morphological types
    #
    # Based on the classification by Choi et al. (2010)
    #---------------------------------------------------------------------------
    # Normal late type
    #---------------------------------------------------------------------------
    normal_late_boolean1 = np.logical_and.reduce((color > 1.8, color < 3.5,
                                                  grad > -0.7, grad < 0.5))
    normal_late_boolean2 = np.logical_and.reduce((color <= 1.8,
                                                  grad > -0.7, grad < 1))

    normal_late_boolean = np.logical_and(late_galaxies_boolean, 
                                         np.logical_or(normal_late_boolean1, 
                                                       normal_late_boolean2))

    # Late-type galaxies in the Green Valley
    gv_normal_late_boolean = np.logical_and.reduce((normal_late_boolean, 
                                                    np.logical_or(aimc <= 135, 
                                                                  aimc >= 315), 
                                                    np.logical_not(gv_early_boolean)))

    # Late-type galaxies in the Blue Cloud
    bc_normal_late_boolean = np.logical_and.reduce((normal_late_boolean, 
                                                    aimc > 135, 
                                                    aimc < 315, 
                                                    np.logical_not(gv_early_boolean)))
    #---------------------------------------------------------------------------
    # Normal early type
    #---------------------------------------------------------------------------
    normal_early_boolean = np.logical_and(early_galaxies_boolean, 
                                          np.logical_and.reduce((color > 2.5, color < 3.5,
                                                                 grad > -0.7,  grad < 1)))
    #---------------------------------------------------------------------------
    # Blue early type
    #---------------------------------------------------------------------------
    blue_early_boolean = np.logical_and(early_galaxies_boolean, 
                                        np.logical_and.reduce((color < 2.5,
                                                               grad > -0.7, grad < 1)))
    ############################################################################



    ############################################################################
    # Color-magnitude classification
    #---------------------------------------------------------------------------
    # Red sequence
    CMDclass[normal_early_boolean] = 3

    # Green valley
    GV_boolean = np.logical_or.reduce((blue_early_boolean, 
                                       gv_early_boolean, 
                                       gv_normal_late_boolean))
    CMDclass[GV_boolean] = 2

    # Blue cloud
    CMDclass[bc_normal_late_boolean] = 1
    ############################################################################

    return CMDclass





################################################################################
################################################################################
################################################################################



def CMD_class(color, grad, cinv, rmag):
    '''
    Determine if a galaxy is in the blue cloud (BC), green valley (GV), or red 
    sequence (RS).

    For use with SDSS photometry; requires:
      - color: u-r
      - color gradient: Delta g-i
      - inverse concentration index


    PARAMETERS
    ==========

    color : list or array of length N
        u-r color of the galaxies

    grad : list or array of length N
        g-i color gradient of the galaxies

    cinv : list or array of length N
        inverse concentration indices of the galaxies

    rmag : list or array of length N
        Apparent Petrosian magnitude of the galaxies


    RETURNS
    =======

    CMDclass : numpy array of length N
        CMD classification for each galaxy
          - 3 = Red sequence
          - 2 = Green valley
          - 1 = Blue cloud
    '''


    ############################################################################
    # Initialize output array
    #---------------------------------------------------------------------------
    CMDclass = -1*np.ones(len(color), dtype=int)
    ############################################################################



    ############################################################################
    # Calculate the aimc values
    #---------------------------------------------------------------------------
    aimc = my_aimc_vertical(grad, color)
    ############################################################################



    ############################################################################
    # Magnitude bins
    #
    # According to Park05, the morphological classification for the galaxies is 
    # slightly different, depending on their apparent magnitudes.
    #---------------------------------------------------------------------------
    boolean_14p5_16 = rmag < 16
    boolean_16_16p5 = np.logical_and(rmag >= 16, rmag < 16.5)
    boolean_16p5_17 = np.logical_and(rmag >= 16.5, rmag < 17)
    boolean_17_17p5 = rmag >= 17
    ############################################################################



    ############################################################################
    # Define the inverse concentration index limit on the intersection of the 
    # skew-normal mixture model
    #
    # Park05 adds a supplemental condition on early-type galaxies based on their 
    # inverse concentration indices.  They used a flat value for each of the 
    # four magnitude bins.  Here, we fit the distribution of the early galaxies 
    # in u-r with a skew-normal mixture model, and define the maximum inverse 
    # concentration index for the early galaxies (the eventual red sequence) 
    # based on the intersection of these two populations.
    #
    # The skew-normal mixture model parameters and intersection points were 
    # found in the morph_classification-2.ipynb Jupyter notebook.
    #---------------------------------------------------------------------------
    cinv_bright = 0.3638991917081808
    cinv_16_16p5 = 0.3769677811861972
    cinv_16p5_17 = 0.38397280636482445
    cinv_faint = 0.3858712301452513
    ############################################################################



    ############################################################################
    # Morphological classification
    #
    # Based on the morphological classification of Park & Choi (2005)
    #---------------------------------------------------------------------------
    # Early type
    #---------------------------------------------------------------------------
    early_boolean1 = np.logical_and(color > 1, grad > 0.3)

    early_boolean2_14p5_16 = np.logical_and.reduce((color > 1, 
                                                    color <= 2.6, 
                                                    grad > -0.28125*color + 0.58125))
    early_boolean2_16_17 = np.logical_and.reduce((color > 1,
                                                  color <= 2.65,
                                                  grad > -0.290909*color + 0.590909))
    early_boolean2_17_17p5 = np.logical_and.reduce((color > 1,
                                                    color <= 2.7,
                                                    grad > -0.28236*color + 0.582353))

    early_boolean3_14p5_16 = np.logical_and(color > 2.60, grad > -0.15)
    early_boolean3_16_16p5 = np.logical_and(color > 2.65, grad > -0.15)
    early_boolean3_16p5_17 = np.logical_and(color > 2.65, grad > -0.25)
    early_boolean3_17_17p5 = np.logical_and(color > 2.70, grad > -0.35)

    early_boolean_14p5_16_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_14p5_16, 
                                                       early_boolean3_14p5_16))
    early_boolean_16_16p5_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_16_17, 
                                                       early_boolean3_16_16p5))
    early_boolean_16p5_17_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_16_17, 
                                                       early_boolean3_16p5_17))
    early_boolean_17_17p5_area = np.logical_or.reduce((early_boolean1, 
                                                       early_boolean2_17_17p5, 
                                                       early_boolean3_17_17p5))

    # Require early-type galaxies to have inverse concentration indices smaller 
    # than some value
    early_boolean_14p5_16 = np.logical_and(early_boolean_14p5_16_area, 
                                           cinv < cinv_bright)
    early_boolean_16_16p5 = np.logical_and(early_boolean_16_16p5_area, 
                                           cinv < cinv_16_16p5)
    early_boolean_16p5_17 = np.logical_and(early_boolean_16p5_17_area, 
                                           cinv < cinv_16p5_17)
    early_boolean_17_17p5 = np.logical_and(early_boolean_17_17p5_area, 
                                           cinv < cinv_faint)

    # Galaxies that would normally have become late-types because of the above 
    # restriction on cinv (conx1) are instead being classified as GV galaxies
    gv_boolean_14p5_16 = np.logical_and(early_boolean_14p5_16_area, 
                                        cinv >= cinv_bright)
    gv_boolean_16_16p5 = np.logical_and(early_boolean_16_16p5_area, 
                                        cinv >= cinv_16_16p5)
    gv_boolean_16p5_17 = np.logical_and(early_boolean_16p5_17_area, 
                                        cinv >= cinv_16p5_17)
    gv_boolean_17_17p5 = np.logical_and(early_boolean_17_17p5_area, 
                                        cinv >= cinv_faint)

    # All early-type galaxies (what will become the red sequence)
    early_galaxies_boolean = np.logical_or.reduce((np.logical_and(boolean_14p5_16, 
                                                                  early_boolean_14p5_16), 
                                                   np.logical_and(boolean_16_16p5, 
                                                                  early_boolean_16_16p5), 
                                                   np.logical_and(boolean_16p5_17, 
                                                                  early_boolean_16p5_17), 
                                                   np.logical_and(boolean_17_17p5, 
                                                                  early_boolean_17_17p5)))
    #---------------------------------------------------------------------------
    # Late type
    #---------------------------------------------------------------------------
    late_boolean_14p5_16 = np.logical_not(early_boolean_14p5_16)
    late_boolean_16_16p5 = np.logical_not(early_boolean_16_16p5)
    late_boolean_16p5_17 = np.logical_not(early_boolean_16p5_17)
    late_boolean_17_17p5 = np.logical_not(early_boolean_17_17p5)
    #---------------------------------------------------------------------------
    late_galaxies_boolean = np.logical_or.reduce((np.logical_and(boolean_14p5_16, 
                                                                 late_boolean_14p5_16),
                                                  np.logical_and(boolean_16_16p5, 
                                                                 late_boolean_16_16p5),
                                                  np.logical_and(boolean_16p5_17, 
                                                                 late_boolean_16p5_17),
                                                  np.logical_and(boolean_17_17p5, 
                                                                 late_boolean_17_17p5)))
    #---------------------------------------------------------------------------
    # Early-type galaxies in the Green Valley
    #---------------------------------------------------------------------------
    gv_early_boolean = np.logical_or.reduce((np.logical_and(boolean_14p5_16, 
                                                            gv_boolean_14p5_16), 
                                             np.logical_and(boolean_16_16p5, 
                                                            gv_boolean_16_16p5), 
                                             np.logical_and(boolean_16p5_17, 
                                                            gv_boolean_16p5_17), 
                                             np.logical_and(boolean_17_17p5, 
                                                            gv_boolean_17_17p5)))
    ############################################################################



    ############################################################################
    # Morphological types
    #
    # Based on the classification by Choi et al. (2010)
    #---------------------------------------------------------------------------
    # Normal late type
    #---------------------------------------------------------------------------
    normal_late_boolean1 = np.logical_and.reduce((color > 1.8, color < 3.5,
                                                  grad > -0.7, grad < 0.5))
    normal_late_boolean2 = np.logical_and.reduce((color <= 1.8,
                                                  grad > -0.7, grad < 1))

    normal_late_boolean = np.logical_and(late_galaxies_boolean, 
                                         np.logical_or(normal_late_boolean1, 
                                                       normal_late_boolean2))

    # Late-type galaxies in the Green Valley
    gv_normal_late_boolean = np.logical_and.reduce((normal_late_boolean, 
                                                    aimc < 110., 
                                                    np.logical_not(gv_early_boolean)))

    # Late-type galaxies in the Blue Cloud
    bc_normal_late_boolean = np.logical_and.reduce((normal_late_boolean, 
                                                    aimc >= 110., 
                                                    np.logical_not(gv_early_boolean)))
    #---------------------------------------------------------------------------
    # Normal early type
    #---------------------------------------------------------------------------
    normal_early_boolean = np.logical_and(early_galaxies_boolean, 
                                          np.logical_and.reduce((color > 2.5, color < 3.5,
                                                                 grad > -0.7,  grad < 1)))
    #---------------------------------------------------------------------------
    # Blue early type
    #---------------------------------------------------------------------------
    blue_early_boolean = np.logical_and(early_galaxies_boolean, 
                                        np.logical_and.reduce((color < 2.5,
                                                               grad > -0.7, grad < 1)))
    ############################################################################



    ############################################################################
    # Color-magnitude classification
    #---------------------------------------------------------------------------
    # Red sequence
    CMDclass[normal_early_boolean] = 3

    # Green valley
    GV_boolean = np.logical_or.reduce((blue_early_boolean, 
                                       gv_early_boolean, 
                                       gv_normal_late_boolean))
    CMDclass[GV_boolean] = 2

    # Blue cloud
    CMDclass[bc_normal_late_boolean] = 1
    ############################################################################

    return CMDclass





################################################################################
################################################################################
################################################################################



def CMD_class_Jan2020(color, grad, cinv):
    '''
    Determine if a galaxy is in the blue cloud (BC), green valley (GV), or red 
    sequence (RS).

    For use with SDSS photometry; requires:
      - color: u-r
      - color gradient: Delta g-i
      - inverse concentration index

    
    PARAMETERS
    ==========

    color : list or array
        Length-N array or list of the u-r color of the galaxies

    grad : list or array
        Length-N array or list of the g-i color gradient of the galaxies

    cinv : list or array
        Length-N array or list of the inverse concentration indices of the 
        galaxies


    RETURNS
    =======
    CMDclass : numpy array
        Length-N array of the CMD classification for each galaxy.
        3 = Red sequence
        2 = Green valley
        1 = Blue cloud
    '''


    ############################################################################
    # Initialize output array
    #---------------------------------------------------------------------------
    CMDclass = -1*np.ones(len(color))
    ############################################################################



    ############################################################################
    # First, red sequence objects
    #---------------------------------------------------------------------------
    RS_boolean1 = np.logical_and.reduce((color > 2.6,
                                         color < 4,
                                         grad > -0.15))
    RS_boolean2 = np.logical_and.reduce((color > 2,
                                         color < 2.6,
                                         grad > -0.28125*color + 0.58125))

    RS_boolean3 = np.logical_or(RS_boolean1, RS_boolean2)

    RS_boolean = np.logical_and(RS_boolean3, cinv < 0.43)

    CMDclass[RS_boolean] = 3
    ############################################################################



    ############################################################################
    # Calculate the aimc values
    #---------------------------------------------------------------------------
    aimc = my_aimc(grad, color)
    ############################################################################



    ############################################################################
    # Next, the green valley objects
    #---------------------------------------------------------------------------
    GV_boolean1 = np.logical_and(RS_boolean3, cinv >= 0.43)

    GV_boolean2 = np.logical_and.reduce((color > 1,
                                         color <=2,
                                         grad > -0.28125*color + 0.58125))

    GV_boolean3 = np.logical_and.reduce((aimc < 20, 
                                         np.logical_not(RS_boolean3),
                                         color < 4))

    GV_boolean = np.logical_or.reduce((GV_boolean1, GV_boolean2, GV_boolean3))

    CMDclass[GV_boolean] = 2
    ############################################################################



    ############################################################################
    # Finally, the blue cloud objects
    #---------------------------------------------------------------------------
    BC_boolean1 = np.logical_and(aimc >= 20, color < 4)

    BC_boolean = np.logical_and.reduce((BC_boolean1,
                                        np.logical_not(GV_boolean),
                                        np.logical_not(RS_boolean)))

    CMDclass[BC_boolean] = 1
    ############################################################################



    return CMDclass





if __name__ == '__main__':

    from astropy.table import Table

    ############################################################################
    # Import data
    #---------------------------------------------------------------------------
    data_directory = '/Users/kellydouglass/Documents/Drexel/Research/Data/'
    data_filename = 'kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag.txt'

    galaxies = Table.read(data_directory + data_filename, 
                          format='ascii.commented_header')
    ############################################################################


    ############################################################################
    # Determine CMD classification
    #---------------------------------------------------------------------------
    galaxies['CMD_class'] = CMD_class(galaxies['u_r'], 
                                      galaxies['cd'], 
                                      galaxies['conx1'], 
                                      galaxies['prmag'])
    ############################################################################


    ############################################################################
    # Save data
    #---------------------------------------------------------------------------
    galaxies.write(data_filename[:-4] + '_CMD.txt', 
                   format='ascii.commented_header', overwrite=True)
    ############################################################################




