import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import io
import glob

def collector(path):
    '''Collects all IDL save data from a given path and stores each file as an element in a list.'''
    
    filenames = glob.glob(path)
    
    data = {'data':[scipy.io.readsav(filenames[i],python_dict=True) \
            for i in range(len(filenames))],'filenames':filenames}
    return data

def separator(data):
    '''Compiles data into separate lists of extended and point sources'''

    point_data = [[data['data'][i]['source_array'][j] \
        for j in range(len(data['data'][i]['source_array'])) \
        if data['data'][i]['source_array'][j][-2] is None ] \
    for i in range(len(data['data']))]
    
    extended_data = [[data['data'][i]['source_array'][j] \
        for j in range(len(data['data'][i]['source_array'])) \
        if data['data'][i]['source_array'][j][-2] is not None ] \
    for i in range(len(data['data'])) ]

    return {'extsources':extended_data,'psources':point_data}

def pixelate(ra_zoom, dec_zoom, n_bins, ra_total, dec_total, flux_total):
    import numpy as np
    #Check to see which dimension is larger so that a square in ra,dec can 
    #be returned
    if (ra_zoom[1]-ra_zoom[0]) > (dec_zoom[1]-dec_zoom[0]):
        zoom = ra_zoom
    else:
        zoom = dec_zoom

    #Find the size of the bins using the largest dimension and the num of bins
    binsize = (zoom[1]-zoom[0])/n_bins

    #Create arrays for ra and dec that give the left side of each pixel
    ra_bin_array = (np.array(range(n_bins)) * binsize) + ra_zoom[0]
    dec_bin_array = (np.array(range(n_bins)) * binsize) + dec_zoom[0]
    #Create an empty array of pixels to be filled in the for loops
    pixels = np.zeros((len(ra_bin_array),len(dec_bin_array)))

    #Histogram components into ra bins
    ra_histogram = np.digitize(ra_total,ra_bin_array)
    ###print ra_histogram

    #Begin for loop over both dimensions of pixels, starting with ra
    for bin_i in range(len(ra_bin_array) - 2):
        ###print range(len(ra_bin_array) -2
        ###print "bin_i",bin_i
        #Find the indices that fall into the current ra bin slice
        ra_inds = np.where(ra_histogram == bin_i)
        ###print "rainds", ra_inds[0]
        ###print "lenrainds", len(ra_inds[0])

        #Go to next for cycle if no indices fall into current ra bin slice
        if len(ra_inds[0]) == 0:
            continue

        #Histogram components that fall into the current ra bin slice by dec
        #print "dectotindex", dec_total[ra_inds]
        #print "decbin", dec_bin_array
        dec_histogram = np.digitize(dec_total[ra_inds],dec_bin_array)
        #print "dechist",dec_histogram
        #Begin for loop by dec over ra bin slice
        for bin_j in range(len(dec_bin_array) -2):
            
            #Find the indicies that fall into the current dec bin
            dec_inds = np.where(dec_histogram == bin_j)

            #Go to next for cycle if no indices fall into current dec bin			
            if len(dec_inds[0]) == 0:
                continue
            #Sum the flux components that fall into current ra/dec bin
            ###print "bi",bin_i,bin_j
            ###print "inds",ra_inds, dec_inds
            pixels[bin_i,bin_j] = np.sum(flux_total[ra_inds[0][dec_inds][0]])

    #Find the pixel centers in ra/dec for plotting purposes
    ra_pixel_centers = (np.arange(n_bins) * binsize) + ra_zoom[0] + binsize/2.
    dec_pixel_centers = (np.arange(n_bins) * binsize) + dec_zoom[0] + binsize/2.

    return pixels, ra_pixel_centers, dec_pixel_centers

def plotall(data,n_bins,minRA,maxRA,minDEC,maxDEC):

    #Separating data into lists of extended and point sources.
    separated = separator(data)
        
    #Creating list of RA coordinates for every point source
    all_point_sources_RA = [separated['psources'][i][j]['RA'] \
        for i in range(len(separated['psources'])) \
        for j in range(len(separated['psources'][i]))]

    #Creating list of DEC coordinates for every point source
    all_point_sources_DEC = [separated['psources'][i][j]['DEC'] \
        for i in range(len(separated['psources'])) \
        for j in range(len(separated['psources'][i]))]

    #Creating list of Stokes I values for every point source
    all_point_sources_I = [separated['psources'][i][j]['FLUX']['I'] \
        for i in range(len(separated['psources'])) \
        for j in range(len(separated['psources'][i]))]

    #Creating list of RA coordinates for every extended source
    all_EO_sources_RA = [separated['extsources'][i][j]['EXTEND']['RA'][k] \
        for i in range(len(separated['extsources'])) \
        for j in range(len(separated['extsources'][i])) \
        for k in range(len(separated['extsources'][i][j]['EXTEND']['RA']))]

    #Creating list of RA coordinates for every extended source
    all_EO_sources_DEC = [separated['extsources'][i][j]['EXTEND']['DEC'][k] \
        for i in range(len(separated['extsources'])) \
        for j in range(len(separated['extsources'][i])) \
        for k in range(len(separated['extsources'][i][j]['EXTEND']['DEC']))]

    #Creating list of Stokes I values for every extended source
    all_EO_sources_I = [separated['extsources'][i][j]['EXTEND']['FLUX'][k]['I'][0] \
        for i in range(len(separated['extsources'])) \
        for j in range(len(separated['extsources'][i])) \
        for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX'])) ]
    
    #Correcting RA to go from -180 degrees to 180 degrees instead of 0 degrees to 360 degrees.
    for i in range(len(all_point_sources_RA)):
        if all_point_sources_RA[i] > 180:
            all_point_sources_RA[i] -= 360
    for i in range(len(all_EO_sources_RA)):
        if all_EO_sources_RA[i] > 180:
            all_EO_sources_RA[i] -= 360
    
    #Combining RA, DEC, and Stokes I values of point and extended sources into comprehensive lists.
    all_RA = all_point_sources_RA + all_EO_sources_RA    
    all_DEC = all_point_sources_DEC + all_EO_sources_DEC
    all_I = all_point_sources_I + all_EO_sources_I
    
    #all_RA_fornax = [all_RA[i] for i in range(len(all_RA)) if ((all_RA[i] > 50 and all_RA[i] < 51.5) and ((all_DEC[i] > -38 and all_DEC[i] < -36.5)))]
    #all_DEC_fornax = [all_DEC[i] for i in range(len(all_DEC)) if ((all_RA[i] > 50 and all_RA[i] < 51.5) and ((all_DEC[i] > -38 and all_DEC[i] < -36.5)))]

    ra_zoom = [min([all_RA[i] for i in range(len(all_RA))\
                if ((all_RA[i] > minRA and all_RA[i] < maxRA)\
                and ((all_DEC[i] > minDEC and all_DEC[i] < maxDEC)))]),\
                max([all_RA[i] for i in range(len(all_RA))\
                if ((all_RA[i] > minRA and all_RA[i] < maxRA)\
                and ((all_DEC[i] > minDEC and all_DEC[i] < maxDEC)))])]
    dec_zoom = [min([all_DEC[i] for i in range(len(all_DEC))\
                if ((all_RA[i] > minRA and all_RA[i] < maxRA)\
                and ((all_DEC[i] > minDEC and all_DEC[i] < maxDEC)))]),\
                max([all_DEC[i] for i in range(len(all_DEC))\
                if ((all_RA[i] > minRA and all_RA[i] < maxRA)\
                and ((all_DEC[i] > minDEC and all_DEC[i] < maxDEC)))])]

    ra_total = np.array(all_RA)
    dec_total = np.array(all_DEC)
    flux_total = np.array(all_I)
    
    (pixels, ra_pixel_centers, dec_pixel_centers) = \
        pixelate(ra_zoom,dec_zoom,n_bins,ra_total,dec_total,flux_total)

    plt.figure(figsize=(9,8))
    plt.imshow(np.transpose(pixels), interpolation = "gaussian", \
        origin = "lower", cmap = matplotlib.cm.get_cmap('afmhot'), \
        extent = [ra_pixel_centers[0], ra_pixel_centers[len(ra_pixel_centers)-1], \
        dec_pixel_centers[0], dec_pixel_centers[len(dec_pixel_centers)-1]])

    plt.colorbar(label='Janskies')
    plt.show()

def plotEO(data,minI,sumI,n_bins):
    
    separated = separator(data)
    
    indexed_EO_sources_RA = [[[separated['extsources'][i][j]['EXTEND']['RA'][k] \
        for k in range(len(separated['extsources'][i][j]['EXTEND']['RA']))] \
            for j in range(len(separated['extsources'][i]))] \
                for i in range(len(separated['extsources'])) ]

    indexed_EO_sources_DEC = [[[separated['extsources'][i][j]['EXTEND']['DEC'][k] \
        for k in range(len(separated['extsources'][i][j]['EXTEND']['DEC']))] \
            for j in range(len(separated['extsources'][i]))] \
                for i in range(len(separated['extsources'])) ]

    indexed_EO_sources_I = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['I'][0] \
        for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX'])) ]
            for j in range(len(separated['extsources'][i]))] \
                for i in range(len(separated['extsources'])) ]
    
    for i in range(len(separated['extsources'])):
        for j in range(len(separated['extsources'][i])):
            if (max(indexed_EO_sources_I[i][j]) > minI) or (sum(indexed_EO_sources_I[i][j]) > sumI):
                ra_zoom = [min(indexed_EO_sources_RA[i][j]),max(indexed_EO_sources_RA[i][j])]
                dec_zoom = [min(indexed_EO_sources_DEC[i][j]),max(indexed_EO_sources_DEC[i][j])]
                
                ra_total = np.array(indexed_EO_sources_RA[i][j])
                dec_total = np.array(indexed_EO_sources_DEC[i][j])
                flux_total = np.array(indexed_EO_sources_I[i][j])
                
                (pixels, ra_pixel_centers, dec_pixel_centers) = \
                pixelate(ra_zoom,dec_zoom,n_bins,ra_total,dec_total,flux_total)
                
                plt.figure(figsize=(9,8))
                plt.imshow(np.transpose(pixels), interpolation = "gaussian", \
                    origin = "lower", cmap = matplotlib.cm.get_cmap('afmhot'), \
                    extent = [ra_pixel_centers[0], ra_pixel_centers[len(ra_pixel_centers)-1], \
                    dec_pixel_centers[0], dec_pixel_centers[len(dec_pixel_centers)-1]])

                plt.title('ObsID: {} at Frequency {}'.format(separated['extsources'][i][j]['ID'], separated['extsources'][i][j]['FREQ']), fontsize = 20)
                plt.xlabel('RA', fontsize = 14)
                plt.ylabel('DEC', fontsize = 14)
                plt.tick_params(size = 8, labelsize = 12)
                plt.minorticks_on()
                plt.tick_params('both', length=12, width=1.8, which='major')
                plt.tick_params('both',length=5, width=1.4, which='minor')
                plt.annotate("ObsID: {}\nTotal Flux: {}\nMean RA: {}\nMean Dec: {}".format \
                    (separated['extsources'][i][j]['ID'], np.sum(indexed_EO_sources_I[i][j]), \
                    np.mean(indexed_EO_sources_RA[i][j]), np.mean(indexed_EO_sources_DEC[i][j])),\
                    xy=(1.5, .5), xytext=(0, 0),\
                    xycoords=('axes fraction', 'figure fraction'),\
                    textcoords='offset points',\
                    size=14, ha='center', va='bottom')
                
                plt.colorbar(label='Janskies')
                plt.savefig('pixelatedEO'+'{}'.format(separated['extsources'][i][j]['ID'])+'.png')
    return plt.show()