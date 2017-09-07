
# coding: utf-8

# In[9]:

import numpy as np
from numpy import inf
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Colormap
import scipy
from scipy import io
import glob
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import aplpy
import clusterer as clst
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import pyvo as vo
from urllib import urlretrieve
import copy
import re

def downloadTGSS(data,EOid,NVSS=False):

    """
    This is used to download a FITS image file of an extended object from TGSS or NVSS. It
    takes a .sav file and an object ID, and outputs the name of the FITS file that is downloaded.
    
    Args:
        data: A .sav file that has been run through clst.collector
        EOid (int): the 5-digit ID describing the extended object (must be from data)
        NVSS (boolean): 
            if True, downloads a FITS file of the object from NVSS
            if False, uses TGSS instead of NVSS (default is False)
            
    Returns: 
        filename: name of the FITS file which is downloaded, based on EOid, RA, and DEC.
    """
    
    from astropy.coordinates import SkyCoord
    import pyvo as vo
    from urllib import urlretrieve

    for i in range(len(data['data'][0]['source_array'])):
        if data['data'][0]['source_array'][i]['id'] == EOid:
            fitsRA = data['data'][0]['source_array'][i]['ra']
            fitsDEC = data['data'][0]['source_array'][i]['dec']
    
    myLocation = SkyCoord(fitsRA*u.deg, fitsDEC*u.deg, frame = 'icrs')


    if NVSS == False:
        query = vo.sia.SIAQuery(
                'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=TGSS&', pos=(myLocation.ra.deg, myLocation.dec.deg),
                size = 0.5, format='image/fits')
        filename = 'EOID{}_RA{}_DEC{}_TGSS.fits'.format(EOid,myLocation.ra.deg, myLocation.dec.deg)
    else:
        query = vo.sia.SIAQuery(
                'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=NVSS&', pos=(myLocation.ra.deg, myLocation.dec.deg),
                size = 0.5, format='image/fits')
        filename = 'EOID{}_RA{}_DEC{}_NVSS.fits'.format(EOid,myLocation.ra.deg, myLocation.dec.deg)
        
    results = query.execute()

    #now we extract the url of the fits file we want
    url = results[0].getdataurl()

    #and download it somewhwere. I’ve hardcoded the name, but you’ll want to have this name change
    urlretrieve(url, filename)
    
    return filename
    
    
def contourFits(data,fits_file,EOid,filename,cluster=False,cutoff=0.03):
    
    from astropy.wcs import WCS
    from astropy.io import fits
    from astropy import units as u
    
    
    '''
    This plots the contours of one extended object over a FITS file of the same object from a 
    different catalog (downloadTGSS can be used to get either TGSS or NVSS FITS files).
    
    Args:
        data: A .sav file that has been run through clst.collector
        fits_file: The FITS file you would like to plot
        filename: name of the image file that this creates
        cluster: boolean
            if True, runs Devin's clustering algorithm on data
            if False, does not 
        cutoff: flux cutoff for clst.modeler, default is 30 milijanskies
        binwidth
        
    Returns:
        filename: name of the image file that this creates
        
    '''
    
    
    separated = clst.separator(data)    
    sought = clst.seeker(data)
    
    if cluster == True:
        modeled = clst.modeler(data,10,0,cutoff=cutoff)

        indexed_EO_sources_RA = copy.deepcopy(modeled['EO_sources_RA'])
        indexed_EO_sources_DEC = copy.deepcopy(modeled['EO_sources_DEC'])
        indexed_EO_sources_FLUX = copy.deepcopy(modeled['EO_sources_I'])
    else:
        indexed_EO_sources_RA = copy.deepcopy(sought['EO_sources_RA'])
        indexed_EO_sources_DEC = copy.deepcopy(sought['EO_sources_DEC'])
        indexed_EO_sources_FLUX = copy.deepcopy(sought['EO_sources_I'])

    binwidth = 2
    sigmawidth=4
    
    binwidth = binwidth * (1. / 3600.) # Converting binwidth in arcsec to degrees
    sigmawidth = sigmawidth * (1. / 3600.) # Converting binwidth in arcsec to degrees
    kernel = Gaussian2DKernel(stddev=(sigmawidth / binwidth))
    semi_zoom = 100. / 60.
    pixelreplacement = 1e-5
    cmap = matplotlib.cm.get_cmap('gist_heat')

    for i in range(len(data['data'])):        
        for j in range(len(separated['extsources'][i])):
            if (separated['extsources'][i][j]['ID'] == EOid):
                
                EO_framed = clst.framer(indexed_EO_sources_RA[i][j],indexed_EO_sources_DEC[i][j])
                EO_RA_zoom = [EO_framed['RA_zoom_min'],EO_framed['RA_zoom_max']]
                EO_DEC_zoom = [EO_framed['DEC_zoom_min'],EO_framed['DEC_zoom_max']]
                EO_RA_total = np.array(indexed_EO_sources_RA[i][j])
                EO_DEC_total = np.array(indexed_EO_sources_DEC[i][j])
                EO_FLUX_total = np.array(indexed_EO_sources_FLUX[i][j])
                EO_n_bins = int(max(EO_RA_zoom[1] - EO_RA_zoom[0],
                                    EO_DEC_zoom[1] - EO_DEC_zoom[0]) / binwidth)
                (EO_pixels, EO_RA_pixel_centers, EO_DEC_pixel_centers) = clst.pixelate(
                    EO_RA_zoom, EO_DEC_zoom, EO_n_bins, EO_RA_total, EO_DEC_total, EO_FLUX_total)
                EO_pixels[EO_pixels == 0] = pixelreplacement
                EO_convolved = convolve(EO_pixels, kernel)
      
                
                meshRA, meshDEC = np.meshgrid(EO_RA_pixel_centers,EO_DEC_pixel_centers)
                RA = meshRA.flatten()
                DEC = meshDEC.flatten()
                DATA = np.transpose(EO_convolved)
                contour_data = DATA.flatten()
                
                w = WCS(fits_file)

                px, py = w.wcs_world2pix(RA,DEC,1)
                
                
                #this is for TGSS contours on MWA graph
                
                meshRA1, meshDEC1 = np.meshgrid(np.arange(269),np.arange(287))
                RA1 = meshRA1.flatten()
                DEC1 = meshDEC1.flatten()
                
                wx, wy = w.wcs_pix2world(RA1,DEC1,1)
                

                fig = plt.figure(figsize=(20,20))
                
                fig.suptitle('EO Source {} from ObsID {}'.format                    
                             (separated['extsources'][i][j]['ID'],                    
                             [int(s) for s in re.findall('\d+',data['filenames'][i])][0],
                             separated['extsources'][i][j]['FREQ']), fontsize = 30)
               
                
                f1 = aplpy.FITSFigure(fits_file,figure=fig,)
                f1.show_colorscale(cmap='hot',vmin=0)   
                f1.show_grid()
                f1.add_colorbar()
                f1.remove_colorbar()
                
                f1.show_scalebar(1)
                f1.scalebar.set_length(2 * u.arcminute)
                f1.scalebar.set_corner(corner='bottom right')
                f1.scalebar.set_label("2 arcminutes")
                f1.scalebar.set_color('blue')
                f1.scalebar.set_frame(True)
                
                
                f1.set_title('TGSS EO Plot with MWA Contours at 0.03 Jansky cutoff (Scale bar = 2 arcminutes)')
                f1.set_xaxis_coord_type('scalar')
                f1.set_yaxis_coord_type('scalar')
                f1.tick_labels.set_xformat('%11.3f')
                f1.tick_labels.set_yformat('%11.3f')
               
                
                ax = fig.gca()

                contours = ax.tricontour(px,py,contour_data,colors='cyan',norm=LogNorm())

                ax.patch.set_alpha(0.0)
                
                plt.savefig(filename, bbox_inches='tight')
                
    return filename


def getFits(data,EOidList,filename,components=True):
    
    '''
    This function takes a clustered .sav file and turns it into a binary FITS file
    
    Args:
        data: a .sav file that has been run through clst.collector
        EOidList: a list of the EO id's for objects which you would like to be clustered. All other objects
                     will not be clustered.
        filename (str): what you would like the FITS file to be names
        components: boolean
            If True: All unclustered EO's are represented by all of their components in the FITS file (default)
            If False: All unclustered EO's are represented by a single point source, with a weighted average
                      of the RA's and DEC's of their components and where flux is the sum of the flux of their
                      components.
                    
    Returns:
        filename (str): what you would like the FITS file to be names
    '''
    
    sought = clst.seeker(data)
    modeled = clst.modeler(data,percent=0,radius=10,cutoff=.03)
    
    names = [int(EOidList[i]) for i, val in enumerate(EOidList)]
    name_indices = [j for j, val in enumerate(sought['EO_sources_ID'][0]) if val[0] in names]
    
    EO_sources_ID = copy.deepcopy(sought['EO_sources_ID'])
    EO_sources_X = copy.deepcopy(sought['EO_sources_X'])
    EO_sources_Y = copy.deepcopy(sought['EO_sources_Y'])
    EO_sources_RA = copy.deepcopy(sought['EO_sources_RA'])
    EO_sources_DEC = copy.deepcopy(sought['EO_sources_DEC'])
    EO_sources_STON = copy.deepcopy(sought['EO_sources_STON'])
    EO_sources_FREQ = copy.deepcopy(sought['EO_sources_FREQ'])
    EO_sources_ALPHA = copy.deepcopy(sought['EO_sources_ALPHA'])
    EO_sources_GAIN = copy.deepcopy(sought['EO_sources_GAIN'])
    EO_sources_FLAG = copy.deepcopy(sought['EO_sources_FLAG'])
    EO_sources_XX = copy.deepcopy(sought['EO_sources_XX'])
    EO_sources_YY = copy.deepcopy(sought['EO_sources_YY'])
    EO_sources_XY = copy.deepcopy(sought['EO_sources_XY'])
    EO_sources_YX = copy.deepcopy(sought['EO_sources_YX'])
    EO_sources_I = copy.deepcopy(sought['EO_sources_I'])
    EO_sources_Q = copy.deepcopy(sought['EO_sources_Q'])
    EO_sources_U = copy.deepcopy(sought['EO_sources_U'])
    EO_sources_V = copy.deepcopy(sought['EO_sources_V'])
    clustered_EO_sources_ID = copy.deepcopy(modeled['EO_sources_ID'])
    clustered_EO_sources_X = copy.deepcopy(modeled['EO_sources_X'])
    clustered_EO_sources_Y = copy.deepcopy(modeled['EO_sources_Y'])
    clustered_EO_sources_RA = copy.deepcopy(modeled['EO_sources_RA'])
    clustered_EO_sources_DEC = copy.deepcopy(modeled['EO_sources_DEC'])
    clustered_EO_sources_STON = copy.deepcopy(modeled['EO_sources_STON'])
    clustered_EO_sources_FREQ = copy.deepcopy(modeled['EO_sources_FREQ'])
    clustered_EO_sources_ALPHA = copy.deepcopy(modeled['EO_sources_ALPHA'])
    clustered_EO_sources_GAIN = copy.deepcopy(modeled['EO_sources_GAIN'])
    clustered_EO_sources_FLAG = copy.deepcopy(modeled['EO_sources_FLAG'])
    clustered_EO_sources_XX = copy.deepcopy(modeled['EO_sources_XX'])
    clustered_EO_sources_YY = copy.deepcopy(modeled['EO_sources_YY'])
    clustered_EO_sources_XY = copy.deepcopy(modeled['EO_sources_XY'])
    clustered_EO_sources_YX = copy.deepcopy(modeled['EO_sources_YX'])
    clustered_EO_sources_I = copy.deepcopy(modeled['EO_sources_I'])
    clustered_EO_sources_Q = copy.deepcopy(modeled['EO_sources_Q'])
    clustered_EO_sources_U = copy.deepcopy(modeled['EO_sources_U'])
    clustered_EO_sources_V = copy.deepcopy(modeled['EO_sources_V'])
    
    #points
    name_index = 0
    if components == True:
        for j, val in enumerate(EO_sources_ID[0]):
            if j in name_indices:
                print "j:", j, "name_index:", name_index
                EO_sources_ID[0][j] = clustered_EO_sources_ID[0][name_indices[name_index]]
                EO_sources_X[0][j] = clustered_EO_sources_X[0][name_indices[name_index]]
                EO_sources_Y[0][j] = clustered_EO_sources_Y[0][name_indices[name_index]]
                EO_sources_RA[0][j] = clustered_EO_sources_RA[0][name_indices[name_index]]
                EO_sources_DEC[0][j] = clustered_EO_sources_DEC[0][name_indices[name_index]]
                EO_sources_STON[0][j] = clustered_EO_sources_STON[0][name_indices[name_index]]
                EO_sources_FREQ[0][j] = clustered_EO_sources_FREQ[0][name_indices[name_index]]
                EO_sources_ALPHA[0][j] = clustered_EO_sources_ALPHA[0][name_indices[name_index]]
                EO_sources_GAIN[0][j] = clustered_EO_sources_GAIN[0][name_indices[name_index]]
                EO_sources_FLAG[0][j] = clustered_EO_sources_FLAG[0][name_indices[name_index]]
                EO_sources_XX[0][j] = clustered_EO_sources_XX[0][name_indices[name_index]]
                EO_sources_YY[0][j] = clustered_EO_sources_YY[0][name_indices[name_index]]
                EO_sources_XY[0][j] = clustered_EO_sources_XY[0][name_indices[name_index]]
                EO_sources_YX[0][j] = clustered_EO_sources_YX[0][name_indices[name_index]]
                EO_sources_I[0][j] = clustered_EO_sources_I[0][name_indices[name_index]]
                EO_sources_Q[0][j] = clustered_EO_sources_Q[0][name_indices[name_index]]
                EO_sources_U[0][j] = clustered_EO_sources_U[0][name_indices[name_index]]
                EO_sources_V[0][j] = clustered_EO_sources_V[0][name_indices[name_index]]
                name_index += 1
            else:
                EO_sources_ID[0][j] = [np.average(EO_sources_ID[0][j], weights=EO_sources_I[0][j])]
                EO_sources_X[0][j] = [np.average(EO_sources_X[0][j], weights=EO_sources_I[0][j])]
                EO_sources_Y[0][j] = [np.average(EO_sources_Y[0][j], weights=EO_sources_I[0][j])]
                EO_sources_RA[0][j] = [np.average(EO_sources_RA[0][j], weights=EO_sources_I[0][j])]
                EO_sources_DEC[0][j] = [np.average(EO_sources_DEC[0][j], weights=EO_sources_I[0][j])]
                EO_sources_STON[0][j] = [np.average(EO_sources_STON[0][j], weights=EO_sources_I[0][j])]
                EO_sources_FREQ[0][j] = [np.average(EO_sources_FREQ[0][j], weights=EO_sources_I[0][j])]
                EO_sources_ALPHA[0][j] = [np.average(EO_sources_ALPHA[0][j], weights=EO_sources_I[0][j])]
                EO_sources_GAIN[0][j] = [np.average(EO_sources_GAIN[0][j], weights=EO_sources_I[0][j])]
                EO_sources_FLAG[0][j] = [np.average(EO_sources_FLAG[0][j], weights=EO_sources_I[0][j])]
                EO_sources_XX[0][j] = [sum(EO_sources_XX[0][j])]
                EO_sources_YY[0][j] = [sum(EO_sources_YY[0][j])]
                EO_sources_XY[0][j] = [sum(EO_sources_XY[0][j])]
                EO_sources_YX[0][j] = [sum(EO_sources_YX[0][j])]
                EO_sources_I[0][j] = [sum(EO_sources_I[0][j])]
                EO_sources_Q[0][j] = [sum(EO_sources_Q[0][j])]
                EO_sources_U[0][j] = [sum(EO_sources_U[0][j])]
                EO_sources_V[0][j] = [sum(EO_sources_V[0][j])]

    elif components == False:
        for j, val in enumerate(EO_sources_ID[0]):
            if j in name_indices:
                print "j:", j, "name_index:", name_index
                EO_sources_ID[0][j] = clustered_EO_sources_ID[0][name_indices[name_index]]
                EO_sources_X[0][j] = clustered_EO_sources_X[0][name_indices[name_index]]
                EO_sources_Y[0][j] = clustered_EO_sources_Y[0][name_indices[name_index]]
                EO_sources_RA[0][j] = clustered_EO_sources_RA[0][name_indices[name_index]]
                EO_sources_DEC[0][j] = clustered_EO_sources_DEC[0][name_indices[name_index]]
                EO_sources_STON[0][j] = clustered_EO_sources_STON[0][name_indices[name_index]]
                EO_sources_FREQ[0][j] = clustered_EO_sources_FREQ[0][name_indices[name_index]]
                EO_sources_ALPHA[0][j] = clustered_EO_sources_ALPHA[0][name_indices[name_index]]
                EO_sources_GAIN[0][j] = clustered_EO_sources_GAIN[0][name_indices[name_index]]
                EO_sources_FLAG[0][j] = clustered_EO_sources_FLAG[0][name_indices[name_index]]
                EO_sources_XX[0][j] = clustered_EO_sources_XX[0][name_indices[name_index]]
                EO_sources_YY[0][j] = clustered_EO_sources_YY[0][name_indices[name_index]]
                EO_sources_XY[0][j] = clustered_EO_sources_XY[0][name_indices[name_index]]
                EO_sources_YX[0][j] = clustered_EO_sources_YX[0][name_indices[name_index]]
                EO_sources_I[0][j] = clustered_EO_sources_I[0][name_indices[name_index]]
                EO_sources_Q[0][j] = clustered_EO_sources_Q[0][name_indices[name_index]]
                EO_sources_U[0][j] = clustered_EO_sources_U[0][name_indices[name_index]]
                EO_sources_V[0][j] = clustered_EO_sources_V[0][name_indices[name_index]]
                name_index += 1
    
    EO_ID = [j for j in EO_sources_ID[0]]
    EO_ID = [val for nest in EO_ID for val in nest]
    EO_X = [j for j in EO_sources_X[0]]
    EO_X = [val for nest in EO_X for val in nest]
    EO_Y = [j for j in EO_sources_Y[0]]
    EO_Y = [val for nest in EO_Y for val in nest]
    EO_RA = [j for j in EO_sources_RA[0]]
    EO_RA = [val for nest in EO_RA for val in nest]
    EO_DEC = [j for j in EO_sources_DEC[0]]
    EO_DEC = [val for nest in EO_DEC for val in nest]
    EO_STON = [j for j in EO_sources_STON[0]]
    EO_STON = [val for nest in EO_STON for val in nest]
    EO_FREQ = [j for j in EO_sources_FREQ[0]]
    EO_FREQ = [val for nest in EO_FREQ for val in nest]
    EO_ALPHA = [j for j in EO_sources_ALPHA[0]]
    EO_ALPHA = [val for nest in EO_ALPHA for val in nest]
    EO_GAIN = [j for j in EO_sources_GAIN[0]]
    EO_GAIN = [val for nest in EO_GAIN for val in nest]
    EO_FLAG = [j for j in EO_sources_FLAG[0]]
    EO_FLAG = [val for nest in EO_FLAG for val in nest]
    EO_XX = [j for j in EO_sources_XX[0]]
    EO_XX = [val for nest in EO_XX for val in nest]
    EO_YY = [j for j in EO_sources_YY[0]]
    EO_YY = [val for nest in EO_YY for val in nest]
    EO_XY = [j for j in EO_sources_XY[0]]
    EO_XY = [val for nest in EO_XY for val in nest]
    EO_YX = [j for j in EO_sources_YX[0]]
    EO_YX = [val for nest in EO_YX for val in nest]
    EO_I = [j for j in EO_sources_I[0]]
    EO_I = [val for nest in EO_I for val in nest]
    EO_Q = [j for j in EO_sources_Q[0]]
    EO_Q = [val for nest in EO_Q for val in nest]
    EO_U = [j for j in EO_sources_U[0]]
    EO_U = [val for nest in EO_U for val in nest]
    EO_V = [j for j in EO_sources_V[0]]
    EO_V = [val for nest in EO_V for val in nest]

   
    col_ID = fits.Column(name='all_sources_ID', format='J', array=EO_ID)

    col_X = fits.Column(name='all_sources_X', format='D', array=EO_X)

    col_Y = fits.Column(name='all_sources_Y', format='D', array=EO_Y)

    col_RA = fits.Column(name='all_sources_RA', format='D', array=EO_RA)

    col_DEC = fits.Column(name='all_sources_RA', format='D', array=EO_DEC)

    col_STON = fits.Column(name='all_sources_STON', format='D', array=EO_STON)

    col_FREQ = fits.Column(name='all_sources_FREQ', format='D', array=EO_FREQ)

    col_ALPHA = fits.Column(name='all_sources_ALPHA', format='D', array=EO_ALPHA)

    col_GAIN = fits.Column(name='all_sources_GAIN', format='D', array=EO_GAIN)

    col_FLAG = fits.Column(name='all_sources_FLAG', format='D', array=EO_FLAG)

    col_XX = fits.Column(name='all_sources_XX', format='D', array=EO_XX)

    col_YY = fits.Column(name='all_sources_YY', format='D', array=EO_YY)

    col_XY = fits.Column(name='all_sources_XY', format='D', array=EO_XY)

    col_YX = fits.Column(name='all_sources_YX', format='D', array=EO_YX)

    col_I = fits.Column(name='all_sources_I', format='D', array=EO_I)

    col_Q = fits.Column(name='all_sources_Q', format='D', array=EO_Q)

    col_U = fits.Column(name='all_sources_U', format='D', array=EO_U)

    col_V = fits.Column(name='all_sources_V', format='D', array=EO_V)


    col_list = []

    col_list.append(col_ID)
    col_list.append(col_X)
    col_list.append(col_Y)
    col_list.append(col_RA)
    col_list.append(col_DEC)
    col_list.append(col_STON)
    col_list.append(col_FREQ)
    col_list.append(col_ALPHA)
    col_list.append(col_GAIN)
    col_list.append(col_FLAG)
    col_list.append(col_XX)
    col_list.append(col_YY)
    col_list.append(col_XY)
    col_list.append(col_YX)
    col_list.append(col_I)
    col_list.append(col_Q)
    col_list.append(col_U)
    col_list.append(col_V)

    
    colDefs = fits.ColDefs(col_list)
    tbhdu = fits.BinTableHDU.from_columns(colDefs)
    
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)

    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(filename)
    
    return filename

