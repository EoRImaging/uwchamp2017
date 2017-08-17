
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
import Clusterer_2 as clst2
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import pyvo as vo
from urllib import urlretrieve

def downloadTGSS(data,EOid,NVSS=False):

    """
    This is used to download a FITS image file of an extended object from TGSS or NVSS
    
    Args:
        data: A .sav file that has been run through clst2.collector
        EOid (int): the 5-digit ID describing the extended object (must be from data)
        NVSS (boolean): 
            if True, downloads a FITS file of the object from NVSS
            if False, uses TGSS instead of NVSS (default is False)
    """
    
    from astropy.coordinates import SkyCoord
    import pyvo as vo
    from urllib import urlretrieve

    for i in range(len(data0['data'][0]['source_array'])):
        if data0['data'][0]['source_array'][i]['id'] == EOid:
            fitsRA = data0['data'][0]['source_array'][i]['ra']
            fitsDEC = data0['data'][0]['source_array'][i]['dec']
    
    myLocation = SkyCoord(fitsRA*u.deg, fitsDEC*u.deg, frame = 'icrs')


    if NVSS == False:
        query = vo.sia.SIAQuery(
                'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=TGSS&', pos=(myLocation.ra.deg, myLocation.dec.deg),
                size = 0.5, format='image/fits')
        filename = 'EOID{}RA{}DEC{}TGSS.fits'.format(EOid,myLocation.ra.deg, myLocation.dec.deg)
    else:
        query = vo.sia.SIAQuery(
                'http://skyview.gsfc.nasa.gov/cgi-bin/vo/sia.pl?survey=NVSS&', pos=(myLocation.ra.deg, myLocation.dec.deg),
                size = 0.5, format='image/fits')
        filename = 'EOID{}RA{}DEC{}NVSS.fits'.format(EOid,myLocation.ra.deg, myLocation.dec.deg)
        
    results = query.execute()

    #now we extract the url of the fits file we want
    url = results[0].getdataurl()

    #and download it somewhwere. I’ve hardcoded the name, but you’ll want to have this name change
    urlretrieve(url, filename)
    
    
def contourFits(data,fits_file,EOid,cluster=False,cutoff=0.03):
    
    from astropy.wcs import WCS
    from astropy.io import fits
    from astropy import units as u
    
    
    '''
    This plots the contours of one extended object over a FITS file of the same object from a 
    different catalog (downloadTGSS can be used to get either TGSS or NVSS FITS files).
    
    Args:
        data: A .sav file that has been run through clst2.collector
        fits_file: The FITS file you would like to plot
        filename: name of the image file that this creates
        cluster: boolean
            if True, runs Devin's clustering algorithm on data
            if False, does not 
        cutoff: flux cutoff for clst.modeler, default is 30 milijanskies
        binwidth
        
        
    '''
    
    
    separated = clst2.separator(data)    
    sought = clst2.seeker(data)
    
    if cluster == True:
        modeled = clst2.modeler(data,10,0,cutoff=cutoff)

        indexed_EO_sources_RA = copy.deepcopy(modeled['EO_sources_RA'])
        indexed_EO_sources_DEC = copy.deepcopy(modeled['EO_sources_DEC'])
        indexed_EO_sources_FLUX = copy.deepcopy(modeled['EO_sources_FLUX'])
    else:
        indexed_EO_sources_RA = copy.deepcopy(sought['EO_sources_RA'])
        indexed_EO_sources_DEC = copy.deepcopy(sought['EO_sources_DEC'])
        indexed_EO_sources_FLUX = copy.deepcopy(sought['EO_sources_FLUX'])

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
                
                EO_framed = framer(indexed_EO_sources_RA[i][j],indexed_EO_sources_DEC[i][j])
                EO_RA_zoom = [EO_framed['RA_zoom_min'],EO_framed['RA_zoom_max']]
                EO_DEC_zoom = [EO_framed['DEC_zoom_min'],EO_framed['DEC_zoom_max']]
                EO_RA_total = np.array(indexed_EO_sources_RA[i][j])
                EO_DEC_total = np.array(indexed_EO_sources_DEC[i][j])
                EO_FLUX_total = np.array(indexed_EO_sources_FLUX[i][j])
                EO_n_bins = int(max(EO_RA_zoom[1] - EO_RA_zoom[0],
                                    EO_DEC_zoom[1] - EO_DEC_zoom[0]) / binwidth)
                (EO_pixels, EO_RA_pixel_centers, EO_DEC_pixel_centers) = pixelate(
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
                
                fig.suptitle('EO Source {} from ObsID {}'.format                    (separated['extsources'][i][j]['ID'],                    [int(s) for s in re.findall('\d+',data['filenames'][i])][0],                    separated['extsources'][i][j]['FREQ']), fontsize = 30)
               
                
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
                
                plt.savefig('modelcutimages/uncutfunk/GLEAM'+'{}'.format                            (separated['extsources'][i][j]['ID'])+'uncutTest.png', bbox_inches='tight')


def getFits(data):
    
    '''
    This function takes a clustered .sav file and turns it into a binary FITS file
    
    Args:
        data: a .sav file that has been run through clst2.collector
    '''
    
    sought = clst2.seeker(data)
    modeled = clst2.modeler(data,percent=0,radius=10,cutoff=.03)
    
    list_ID = []
    list_X = []
    list_Y = []
    list_RA = []
    list_DEC = []
    list_STON = []
    list_FREQ = []
    list_ALPHA = []
    list_GAIN = []
    list_FLAG = []
    list_XX = []
    list_YY = []
    list_XY = []
    list_YX = []
    list_I = []
    list_Q = []
    list_U = []
    list_V = []


    EO_parents_len = len(mod['EO_sources_RA'][0])
    point_source_len = len(cls['point_sources_DEC'][0])


    for i in range(EO_parents_len):
        for j in range(len(mod['EO_sources_RA'][0][i])):
            list_ID.append(mod['EO_sources_ID'][0][i][j])
            list_X.append(mod['EO_sources_X'][0][i][j])
            list_Y.append(mod['EO_sources_Y'][0][i][j])
            list_RA.append(mod['EO_sources_RA'][0][i][j])
            list_DEC.append(mod['EO_sources_DEC'][0][i][j])
            list_STON.append(mod['EO_sources_STON'][0][i][j])
            list_FREQ.append(mod['EO_sources_FREQ'][0][i][j])
            list_ALPHA.append(mod['EO_sources_ALPHA'][0][i][j])
            list_GAIN.append(mod['EO_sources_GAIN'][0][i][j])
            list_FLAG.append(mod['EO_sources_FLAG'][0][i][j])
            list_XX.append(mod['EO_sources_XX'][0][i][j])
            list_YY.append(mod['EO_sources_YY'][0][i][j])
            list_XY.append(mod['EO_sources_XY'][0][i][j])
            list_YX.append(mod['EO_sources_YX'][0][i][j])
            list_I.append(mod['EO_sources_I'][0][i][j])
            list_Q.append(mod['EO_sources_Q'][0][i][j])
            list_U.append(mod['EO_sources_U'][0][i][j])
            list_V.append(mod['EO_sources_V'][0][i][j])

    for i in range(point_source_len):
        list_ID.append(cls['point_sources_ID'][0][i])
        list_X.append(cls['point_sources_X'][0][i])
        list_Y.append(cls['point_sources_Y'][0][i])
        list_RA.append(cls['point_sources_RA'][0][i])
        list_DEC.append(cls['point_sources_DEC'][0][i])
        list_STON.append(cls['point_sources_STON'][0][i])
        list_FREQ.append(cls['point_sources_FREQ'][0][i])
        list_ALPHA.append(cls['point_sources_ALPHA'][0][i])
        list_GAIN.append(cls['point_sources_GAIN'][0][i])
        list_FLAG.append(cls['point_sources_FLAG'][0][i])
        list_XX.append(cls['point_sources_XX'][0][i])
        list_YY.append(cls['point_sources_YY'][0][i])
        list_XY.append(cls['point_sources_XY'][0][i])
        list_YX.append(cls['point_sources_YX'][0][i])
        list_I.append(cls['point_sources_I'][0][i])
        list_Q.append(cls['point_sources_Q'][0][i])
        list_U.append(cls['point_sources_U'][0][i])
        list_V.append(cls['point_sources_V'][0][i])
        
        
    col_ID = fits.Column(name='all_sources_ID', format='J',        array=list_ID)

    col_X = fits.Column(name='all_sources_X', format='D',            array=list_X)

    col_Y = fits.Column(name='all_sources_Y', format='D',            array=list_Y)

    col_RA = fits.Column(name='all_sources_RA', format='D',            array=list_RA)

    col_DEC = fits.Column(name='all_sources_RA', format='D',            array=list_DEC)

    col_STON = fits.Column(name='all_sources_STON', format='D',            array=list_STON)

    col_FREQ = fits.Column(name='all_sources_FREQ', format='D',            array=list_FREQ)

    col_ALPHA = fits.Column(name='all_sources_ALPHA', format='D',            array=list_ALPHA)

    col_GAIN = fits.Column(name='all_sources_GAIN', format='D',            array=list_GAIN)

    col_FLAG = fits.Column(name='all_sources_FLAG', format='D',            array=list_FLAG)

    col_XX = fits.Column(name='all_sources_XX', format='D',            array=list_XX)

    col_YY = fits.Column(name='all_sources_YY', format='D',            array=list_YY)

    col_XY = fits.Column(name='all_sources_XY', format='D',            array=list_XY)

    col_YX = fits.Column(name='all_sources_YX', format='D',            array=list_YX)

    col_I = fits.Column(name='all_sources_I', format='D',            array=list_I)

    col_Q = fits.Column(name='all_sources_Q', format='D',            array=list_Q)

    col_U = fits.Column(name='all_sources_U', format='D',            array=list_U)

    col_V = fits.Column(name='all_sources_V', format='D',            array=list_V)


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

