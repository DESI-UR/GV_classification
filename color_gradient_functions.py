import numpy as np
from astropy.table import Table

from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300

# softening param for asinh mags

b_dict = {'u': 1.4e-10,
          'g': 0.9e-10,
          'r': 1.2e-10,
          'i': 1.8e-10,
          'z':7.4e-10,}



def SGA_cog(r, mtot, m0, a1, a2):
    '''
    curve of growth from the Siena Galaxy Atlas

    PARAMETERS
    ==========

    r : float
        radius in arcsec

    mtot, m0, a1, a2 : float
        parameters of the fit

    RETURNS
    =======

    m : float
        magnitude within radius r
    
    
    '''

    m = mtot + m0 * np.log(1 + a1*(r/10)**(-a2))

    return m



def asinh_mag_from_flux(flux, sdss_filter):
    '''

    calculate asinh magnitude given flux

    PARAMETERS
    ==========

    flux : float
        value of the flux in maggies

    sdss_filter : string
        'u', 'g', 'r', 'i', or 'z'

    RETURNS
    =======

    m : float
        asinh magnitude
    
    '''

    b = b_dict[sdss_filter]

    m = -2.5/np.log(10) * (np.arcsinh(flux/(2*b)) + np.log(b))

    return m

def pogson_mag_from_flux(flux):
    '''

    calculate asinh magnitude given flux

    PARAMETERS
    ==========

    flux : float
        value of the flux in nanomaggies

    RETURNS
    =======

    m : float
        pogson magnitude

    '''


    m = 22.5 -2.5*np.log10(flux)

    return m


def flux_from_pogson_mag(mag):

    '''
    calculate flux from pogson magnitude

    PARAMETERS
    ==========

    mag : float
        pogson magnitude

    RETURNS
    =======

    flux : float
        flux in nanomaggies
    
    '''

    f = 10**(-mag/2.5) * 10**9

    return f


def extract_cog_data(iauname, sdss_filter, DATA_FOLDER):

    '''
    
    extract curve of growth files

    PARAMETERS
    ==========

    iauname : string
        iauname for galaxy

    sdss_filter : string
        sdss filter 'u', 'g', 'r', 'i', or 'z'
    
    DATA_FOLDER : string
        location of curve of growth files

    RETURNS
    =======

    rs : array
        radii in arcsec

    fs : array
        fluxes at rs in nanomaggies

    
    '''

    fn = DATA_FOLDER + '/' + iauname + '-8-' + sdss_filter + '-cog.fits'

    try:

        cog_table = Table.read(fn, format='fits')

    except:
        print('File not found ', iauname)
        return None, None

    rs = np.array(cog_table['rbins']*cog_table['pixscale'])[0]

    fs = np.array(cog_table['fbins'])[0]

    return rs, fs


def plot_curve_of_growth(i_radius, i_flux, g_radius, g_flux, i_mag_fit, 
                         g_mag_fit, iauname, PLOT_DIR):
    
    
    # plot with residual

    fig, ax = plt.subplots(2,1, height_ratios = (2,1), figsize=(5,6), 
                           sharex=True)

    ax[0].scatter(g_radius, g_flux, 
                  marker='.', label='$g-$band', color='darkgreen',alpha=0.5)
    ax[0].scatter(i_radius, i_flux, 
                  marker='.', label='$i-$band', color='tab:red', alpha=0.5)
    

    g_model = 10**((SGA_cog(g_radius, g_mag_fit[0], g_mag_fit[1], g_mag_fit[2],
                      g_mag_fit[3]) -22.5)/-2.5)
    
    i_model = 10**((SGA_cog(i_radius, i_mag_fit[0], i_mag_fit[1], i_mag_fit[2],
                      i_mag_fit[3])-22.5)/-2.5)
    


    ax[0].plot(g_radius, g_model, color='limegreen')
    ax[0].plot(i_radius, i_model, color='darkorange')

    i_res = i_model - i_flux
    g_res = g_model - g_flux

    ax[1].plot(i_radius, i_res, color='darkorange')
    ax[1].plot(g_radius, g_res, color='limegreen')

    ax[1].axhline(0, linestyle='--', color='k')

    y_ext = np.max([np.max(np.abs(i_res)), np.max(np.abs(g_res))])

    ax[1].set_ylim(-y_ext-3, y_ext+3)

    ax[0].legend()

    ax[1].set_xlabel('radius [arcsec]', fontsize=12)
    ax[1].set_ylabel('residual', fontsize=12)
    ax[0].set_ylabel('flux [nMgy]', fontsize=12)

    ax[0].set_title(iauname + ' curve of growth', fontsize=14)

    fig.savefig(PLOT_DIR + '/' + iauname + '.png', bbox_inches='tight')

    plt.close(fig)




def calculate_color_gradient(i_flux, g_flux, i_radius, g_radius, Rpet, 
                             iauname, PLOT_DIR):
    

    '''
    calculate color gradient by fitting curve of growth curves w SGA model

    PARAMETERS
    ==========

    i_flux : array
        i-band flux in nMgy

    g_flux : array
        g-band flux in nMgy

    i_radius : array
        i-band radii in arcsec

    g_radius : array
        g-band radii in arcsec

    Rpet : float
        petrosian radius in arcsec

    iauname : str
        galaxy iauname

    PLOT_DIR : str
        location to save plots

    
    RETURNS
    =======

    i_mag_fit : array
        best fit params for i band curve of growth
    
    i_mag_cov: array
        i-band fit covariance matrix 
        
    g_mag_fit : array
        best fit aprams for g band curve of growth 
        
    g_mag_cov : array
        g-band fit covariance matrix
        
    cd : float
        color gradient
    
    '''

    # convert i and g fluxes to pogson magnitudes

    i_mag = pogson_mag_from_flux(i_flux)
    g_mag = pogson_mag_from_flux(g_flux)

    # fit curve of growth in magnitude space

    i_mag_fit, i_mag_cov = curve_fit(SGA_cog, 
                                     i_radius, 
                                     i_mag)
    
    g_mag_fit, g_mag_cov = curve_fit(SGA_cog, 
                                     g_radius, 
                                     g_mag)
    
    # plot cog
    
    plot_curve_of_growth(i_radius, i_flux, g_radius, g_flux, i_mag_fit,
                         g_mag_fit, iauname, PLOT_DIR)

    # magnitude within 0.5Rpet

    i_in_mag = SGA_cog(0.5*Rpet, i_mag_fit[0], i_mag_fit[1], 
                       i_mag_fit[2], i_mag_fit[3])
    
    g_in_mag = SGA_cog(0.5*Rpet, g_mag_fit[0], g_mag_fit[1], 
                       g_mag_fit[2], g_mag_fit[3])

    
    # magnitude within Rpet

    i_Rpet_mag = SGA_cog(Rpet, i_mag_fit[0], i_mag_fit[1], 
                         i_mag_fit[2], i_mag_fit[3])
    
    g_Rpet_mag = SGA_cog(Rpet, g_mag_fit[0], g_mag_fit[1], 
                         g_mag_fit[2], g_mag_fit[3])
    

    # convert back to flux

    i_in_flux = 10**((22.5 - i_in_mag)/2.5)
    i_Rpet_flux = 10**((22.5 - i_Rpet_mag)/2.5) 


    g_in_flux = 10**((22.5 - g_in_mag)/2.5)
    g_Rpet_flux = 10**((22.5 - g_Rpet_mag)/2.5) 

    # get flux in annulus

    i_out_flux = i_Rpet_flux - i_in_flux
    g_out_flux = g_Rpet_flux - g_in_flux

    # now calculate color gradient

    delta_i = -2.5 * np.log10(i_out_flux/i_in_flux)

    delta_g = -2.5*np.log10(g_out_flux/g_in_flux)

    cd = delta_g - delta_i

    return i_mag_fit, i_mag_cov, g_mag_fit, g_mag_cov, cd