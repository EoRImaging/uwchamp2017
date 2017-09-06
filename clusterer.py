import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colorbar import Colorbar
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab

import scipy
from scipy import io
from scipy.stats import iqr, norm
from scipy.stats.kde import gaussian_kde

from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

import copy
import glob
import re


def collector(path):
    '''
    The collector reads one or multiple IDL .sav files
    and appends each .sav file's entire data structure to a list.
    
    Args:
        path: The file system location of an IDL .sav file
            or a folder of IDL .sav files.

    Returns:
        data: A dictonary containing two keys:
            1. "data": The data structure of the IDL .sav file(s).
            2. "filenames": The filename associated with the IDL .sav file(s).
        
    '''

    # Glob module finds all the pathnames matching a specified pattern
    # according to the rules used by the Unix shell.
    filenames = glob.glob(path)

    # Filenames are read into python with the scipy.io.readsav function.
    # Data structures and filenames and appended to their own lists
    # with dictionary keys.
    data = {'data': [scipy.io.readsav(filenames[i], python_dict=True)
            for i in range(len(filenames))], 'filenames': filenames}

    # An example for accessing the data structure from the
    # first or only .sav file read in by collector:
    # data[0]['data']
    #
    # An example for accessing the filename from the
    # fourth .sav file read in by collector:
    # data[3]['filenames']

    return data


def separator(data):
    '''
    The separator splits all data which has been read in by the collector
    into individual lists for point sources and extended sources.
    
    This is primarily written to be used by the Seeker.
    
    Args:
        data: The variable assigned to the IDL .sav file which has been
            read in by the collector.

    Returns:
        data: A dictonary containing two keys:
            1. "extsources": The same data structure containing only extended sources.
            2. "psources": The same data structure containing only point sources.
    '''

    # A source is determined to be a point source if
    # it does not contain any extended components.
    point_data = [[data['data'][i]['source_array'][j]
                   for j in range(len(data['data'][i]['source_array']))
                  if data['data'][i]['source_array'][j]['EXTEND'] is None]
                  for i in range(len(data['data']))]

    # A source is determined to be an extended source if
    # it does contain any extended components.
    extended_data = [[data['data'][i]['source_array'][j]
                      for j in range(len(data['data'][i]['source_array']))
                      if data['data'][i]['source_array'][j]['EXTEND']
                      is not None]
                     for i in range(len(data['data']))]

    return {'extsources': extended_data, 'psources': point_data}


def pixelate(ra_zoom, dec_zoom, n_bins, ra_total, dec_total, flux_total):
    import numpy as np
    # Check to see which dimension is larger so that
    # a square in RA and DEC can be returned.
    if (ra_zoom[1] - ra_zoom[0]) > (dec_zoom[1] - dec_zoom[0]):
        zoom = ra_zoom
    else:
        zoom = dec_zoom

    # Find the size of the bins using the largest dimension
    # and the number of bins.
    binsize = (zoom[1] - zoom[0]) / n_bins

    # Create arrays for RA and DEC that give the left side of each pixel
    ra_bin_array = (np.array(range(n_bins)) * binsize) + ra_zoom[0]
    dec_bin_array = (np.array(range(n_bins)) * binsize) + dec_zoom[0]

    # Create an empty array of pixels to be filled in the for loops
    pixels = np.zeros((len(ra_bin_array), len(dec_bin_array)))

    # Histogram components into ra bins
    ra_histogram = np.digitize(ra_total, ra_bin_array)

    # Begin for loop over both dimensions of pixels, starting with RA
    for bin_i in range(len(ra_bin_array) - 2):

        # Find the indices that fall into the current RA bin slice
        ra_inds = np.where(ra_histogram == bin_i)

        # Go to next for cycle if no indices fall into current RA bin slice
        if len(ra_inds[0]) == 0:
            continue

        # Histogram components that fall into the current RA bin slice by DEC
        dec_histogram = np.digitize(dec_total[ra_inds], dec_bin_array)

        # Begin for loop by DEC over RA bin slice
        for bin_j in range(len(dec_bin_array) - 2):

            # Find the indicies that fall into the current DEC bin
            dec_inds = np.where(dec_histogram == bin_j)

            # Go to next for cycle if no indices fall into current DEC bin
            if len(dec_inds[0]) == 0:
                continue

            # Sum the flux components that fall into current RA/DEC bin.
            pixels[bin_i, bin_j] = np.sum(flux_total[ra_inds[0][dec_inds][0]])

    # Find the pixel centers in RA/DEC for plotting purposes.
    ra_pixel_centers = (np.arange(n_bins) * binsize) + ra_zoom[0] + binsize / 2.
    dec_pixel_centers = (np.arange(n_bins) * binsize) + dec_zoom[0] + binsize / 2.

    return pixels, ra_pixel_centers, dec_pixel_centers


def seeker(data):
    """
    The seeker splits all data which has been read in by the collector
    into individual lists for RA, DEC, FLUX, XX, and YY values.
    Separate lists are made for point sources, extended sources,
    and all sources.
    
    Args:
        data: The variable assigned to the IDL .sav file which has been
            read in by the collector.

    Returns:
        Separated lists for ID, X, Y, RA, DEC, STON, FREQ, ALPHA, GAIN, FLAG,
        XX, YY, XY, YX, I, Q, U, and V values for point sources, extended sources,
        and all sources.
    """

    # Separating data into point sources and extended sources.
    separated = separator(data)

    # Creating individual lists for RA, DEC, FLUX, XX, and YY values
    # for point sources.
    point_sources_ID = [[separated['psources'][i][j]['ID']
                         for j in range(len(separated['psources'][i]))]
                        for i in range(len(separated['psources']))]
    point_sources_X = [[separated['psources'][i][j]['X']
                        for j in range(len(separated['psources'][i]))]
                       for i in range(len(separated['psources']))]
    point_sources_Y = [[separated['psources'][i][j]['Y']
                        for j in range(len(separated['psources'][i]))]
                       for i in range(len(separated['psources']))]
    point_sources_RA = [[separated['psources'][i][j]['RA']
                         for j in range(len(separated['psources'][i]))]
                        for i in range(len(separated['psources']))]
    point_sources_DEC = [[separated['psources'][i][j]['DEC']
                          for j in range(len(separated['psources'][i]))]
                         for i in range(len(separated['psources']))]
    point_sources_STON = [[separated['psources'][i][j]['STON']
                           for j in range(len(separated['psources'][i]))]
                          for i in range(len(separated['psources']))]
    point_sources_FREQ = [[separated['psources'][i][j]['FREQ']
                           for j in range(len(separated['psources'][i]))]
                          for i in range(len(separated['psources']))]
    point_sources_ALPHA = [[separated['psources'][i][j]['ALPHA']
                            for j in range(len(separated['psources'][i]))]
                           for i in range(len(separated['psources']))]
    point_sources_GAIN = [[separated['psources'][i][j]['GAIN']
                           for j in range(len(separated['psources'][i]))]
                          for i in range(len(separated['psources']))]
    point_sources_FLAG = [[separated['psources'][i][j]['FLAG']
                           for j in range(len(separated['psources'][i]))]
                          for i in range(len(separated['psources']))]
    point_sources_XX = [[separated['psources'][i][j]['FLUX']['XX'][0]
                         for j in range(len(separated['psources'][i]))]
                        for i in range(len(separated['psources']))]
    point_sources_YY = [[separated['psources'][i][j]['FLUX']['YY'][0]
                         for j in range(len(separated['psources'][i]))]
                        for i in range(len(separated['psources']))]
    point_sources_XY = [[separated['psources'][i][j]['FLUX']['XY'][0]
                         for j in range(len(separated['psources'][i]))]
                        for i in range(len(separated['psources']))]
    point_sources_YX = [[separated['psources'][i][j]['FLUX']['YX'][0]
                         for j in range(len(separated['psources'][i]))]
                        for i in range(len(separated['psources']))]
    point_sources_I = [[separated['psources'][i][j]['FLUX']['I'][0]
                        for j in range(len(separated['psources'][i]))]
                       for i in range(len(separated['psources']))]
    point_sources_Q = [[separated['psources'][i][j]['FLUX']['Q'][0]
                        for j in range(len(separated['psources'][i]))]
                       for i in range(len(separated['psources']))]
    point_sources_U = [[separated['psources'][i][j]['FLUX']['U'][0]
                        for j in range(len(separated['psources'][i]))]
                       for i in range(len(separated['psources']))]
    point_sources_V = [[separated['psources'][i][j]['FLUX']['V'][0]
                        for j in range(len(separated['psources'][i]))]
                       for i in range(len(separated['psources']))]

    # Creating individual lists for RA, DEC, FLUX, XX, and YY values
    # for extended sources.
    EO_sources_ID = [[[separated['extsources'][i][j]['EXTEND']['ID'][k]
                       for k in range(len(separated['extsources'][i][j]['EXTEND']['ID']))]
                      for j in range(len(separated['extsources'][i]))]
                     for i in range(len(separated['extsources']))]
    EO_sources_X = [[[separated['extsources'][i][j]['EXTEND']['X'][k]
                      for k in range(len(separated['extsources'][i][j]['EXTEND']['X']))]
                     for j in range(len(separated['extsources'][i]))]
                    for i in range(len(separated['extsources']))]
    EO_sources_Y = [[[separated['extsources'][i][j]['EXTEND']['Y'][k]
                      for k in range(len(separated['extsources'][i][j]['EXTEND']['Y']))]
                     for j in range(len(separated['extsources'][i]))]
                    for i in range(len(separated['extsources']))]
    EO_sources_RA = [[[separated['extsources'][i][j]['EXTEND']['RA'][k]
                       for k in range(len(separated['extsources'][i][j]['EXTEND']['RA']))]
                      for j in range(len(separated['extsources'][i]))]
                     for i in range(len(separated['extsources']))]
    EO_sources_DEC = [[[separated['extsources'][i][j]['EXTEND']['DEC'][k]
                        for k in range(len(separated['extsources'][i][j]['EXTEND']['DEC']))]
                       for j in range(len(separated['extsources'][i]))]
                      for i in range(len(separated['extsources']))]
    EO_sources_STON = [[[separated['extsources'][i][j]['EXTEND']['STON'][k]
                         for k in range(len(separated['extsources'][i][j]['EXTEND']['STON']))]
                        for j in range(len(separated['extsources'][i]))]
                       for i in range(len(separated['extsources']))]
    EO_sources_FREQ = [[[separated['extsources'][i][j]['EXTEND']['FREQ'][k]
                         for k in range(len(separated['extsources'][i][j]['EXTEND']['FREQ']))]
                        for j in range(len(separated['extsources'][i]))]
                       for i in range(len(separated['extsources']))]
    EO_sources_ALPHA = [[[separated['extsources'][i][j]['EXTEND']['ALPHA'][k]
                          for k in range(len(separated['extsources'][i][j]['EXTEND']['ALPHA']))]
                         for j in range(len(separated['extsources'][i]))]
                        for i in range(len(separated['extsources']))]
    EO_sources_GAIN = [[[separated['extsources'][i][j]['EXTEND']['GAIN'][k]
                         for k in range(len(separated['extsources'][i][j]['EXTEND']['GAIN']))]
                        for j in range(len(separated['extsources'][i]))]
                       for i in range(len(separated['extsources']))]
    EO_sources_FLAG = [[[separated['extsources'][i][j]['EXTEND']['FLAG'][k]
                         for k in range(len(separated['extsources'][i][j]['EXTEND']['FLAG']))]
                        for j in range(len(separated['extsources'][i]))]
                       for i in range(len(separated['extsources']))]
    EO_sources_XX = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['XX'][0]
                       for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                      for j in range(len(separated['extsources'][i]))]
                     for i in range(len(separated['extsources']))]
    EO_sources_YY = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['YY'][0]
                       for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                      for j in range(len(separated['extsources'][i]))]
                     for i in range(len(separated['extsources']))]
    EO_sources_XY = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['XY'][0]
                       for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                      for j in range(len(separated['extsources'][i]))]
                     for i in range(len(separated['extsources']))]
    EO_sources_YX = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['YX'][0]
                       for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                      for j in range(len(separated['extsources'][i]))]
                     for i in range(len(separated['extsources']))]
    EO_sources_I = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['I'][0]
                      for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                     for j in range(len(separated['extsources'][i]))]
                    for i in range(len(separated['extsources']))]
    EO_sources_Q = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['Q'][0]
                      for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                     for j in range(len(separated['extsources'][i]))]
                    for i in range(len(separated['extsources']))]
    EO_sources_U = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['U'][0]
                      for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                     for j in range(len(separated['extsources'][i]))]
                    for i in range(len(separated['extsources']))]
    EO_sources_V = [[[separated['extsources'][i][j]['EXTEND']['FLUX'][k]['V'][0]
                      for k in range(len(separated['extsources'][i][j]['EXTEND']['FLUX']))]
                     for j in range(len(separated['extsources'][i]))]
                    for i in range(len(separated['extsources']))]

    # Fixing RA values to range between -180 and +180.
    for i in range(len(data['data'])):
        for j in range(len(EO_sources_RA[i])):
            for k in range(len(EO_sources_RA[i][j])):
                if EO_sources_RA[i][j][k] > 180:
                    EO_sources_RA[i][j][k] -= 360
        for j in range(len(point_sources_RA[i])):
            if point_sources_RA[i][j] > 180:
                point_sources_RA[i][j] -= 360

    # Adding point source and extended source lists to create
    # lists for RA, DEC, FLUX, XX, and YY values for all sources.
    all_RA = [[point_sources_RA[i][j]
               for j in range(len(point_sources_RA[i]))] +
              [EO_sources_RA[i][j][k]
              for j in range(len(EO_sources_RA[i]))
              for k in range(len(EO_sources_RA[i][j]))]
              for i in range(len(data['data']))]
    all_DEC = [[point_sources_DEC[i][j]
                for j in range(len(point_sources_DEC[i]))] +
               [EO_sources_DEC[i][j][k]
               for j in range(len(EO_sources_DEC[i]))
               for k in range(len(EO_sources_DEC[i][j]))]
               for i in range(len(data['data']))]
    all_I = [[point_sources_I[i][j]
              for j in range(len(point_sources_I[i]))] +
             [EO_sources_I[i][j][k]
             for j in range(len(EO_sources_I[i]))
             for k in range(len(EO_sources_I[i][j]))]
             for i in range(len(data['data']))]
    all_XX = [[point_sources_XX[i][j]
               for j in range(len(point_sources_XX[i]))] +
              [EO_sources_XX[i][j][k]
              for j in range(len(EO_sources_XX[i]))
              for k in range(len(EO_sources_XX[i][j]))]
              for i in range(len(data['data']))]
    all_YY = [[point_sources_YY[i][j]
               for j in range(len(point_sources_YY[i]))] +
              [EO_sources_YY[i][j][k]
              for j in range(len(EO_sources_YY[i]))
              for k in range(len(EO_sources_YY[i][j]))]
              for i in range(len(data['data']))]
    all_BEAM = [np.asarray(np.asarray(all_XX[i]) +
                np.asarray(all_YY[i])) / np.asarray(all_I[i])
                for i in range(len(data['data']))]

    return {'point_sources_ID': point_sources_ID,
            'point_sources_X': point_sources_X,
            'point_sources_Y': point_sources_Y,
            'point_sources_RA': point_sources_RA,
            'point_sources_DEC': point_sources_DEC,
            'point_sources_STON': point_sources_STON,
            'point_sources_FREQ': point_sources_FREQ,
            'point_sources_ALPHA': point_sources_ALPHA,
            'point_sources_GAIN': point_sources_GAIN,
            'point_sources_FLAG': point_sources_FLAG,
            'point_sources_XX': point_sources_XX,
            'point_sources_YY': point_sources_YY,
            'point_sources_XY': point_sources_XY,
            'point_sources_YX': point_sources_YX,
            'point_sources_I': point_sources_I,
            'point_sources_Q': point_sources_Q,
            'point_sources_U': point_sources_U,
            'point_sources_V': point_sources_V,
            'EO_sources_ID': EO_sources_ID,
            'EO_sources_X': EO_sources_X,
            'EO_sources_Y': EO_sources_Y,
            'EO_sources_RA': EO_sources_RA,
            'EO_sources_DEC': EO_sources_DEC,
            'EO_sources_STON': EO_sources_STON,
            'EO_sources_FREQ': EO_sources_FREQ,
            'EO_sources_ALPHA': EO_sources_ALPHA,
            'EO_sources_GAIN': EO_sources_GAIN,
            'EO_sources_FLAG': EO_sources_FLAG,
            'EO_sources_XX': EO_sources_XX,
            'EO_sources_YY': EO_sources_YY,
            'EO_sources_XY': EO_sources_XY,
            'EO_sources_YX': EO_sources_YX,
            'EO_sources_I': EO_sources_I,
            'EO_sources_Q': EO_sources_Q,
            'EO_sources_U': EO_sources_U,
            'EO_sources_V': EO_sources_V,
            'all_RA': all_RA, 'all_DEC': all_DEC, 'all_I': all_I,
            'all_XX': all_XX, 'all_YY': all_YY, 'all_BEAM': all_BEAM}


def chaser(EO_sources_ID, EO_sources_X, EO_sources_Y,
           EO_sources_RA, EO_sources_DEC, EO_sources_STON,
           EO_sources_FREQ, EO_sources_ALPHA, EO_sources_GAIN,
           EO_sources_FLAG, EO_sources_XX, EO_sources_YY,
           EO_sources_XY, EO_sources_YX, EO_sources_I,
           EO_sources_Q, EO_sources_U, EO_sources_V,
           radius, i=0, j=0, n=0):
    """
    The chaser finds the index of the brightest component of an extended object.
    Then, it searches for all neighboring components within a user-defined radius in arcseconds,
    and returns them in a list.
    
    This is primarily written to be used by the Clusterer.
    
    Args:
        EO_sources_*: Lists for an EO's ID, X, Y, RA, DEC, STON, FREQ, ALPHA, GAIN, FLAG,
            XX, YY, XY, YX, I, Q, U, and V as returned by the Seeker.
        radius: A user defined radius to collect neighboring components.
        i: The index of the IDL .sav file which has been read in by the Collector
            (if only one IDL .sav file is being read in, this will be 0).
        j: The index of the extended object.
        n: A variable that limits the amount of indices for which a brightest component can be found.
            For example: if n=3, then the Chaser will search all indices except for the last 3
            to find the EO's brightest component. This is an important addition because the Clusterer appends
            clustered components to the end of the EO_sources_* lists. So having an n-value restricts the Clusterer
            from redundantly choosing a the same component it has already clustered while it loops
            to find other bright components.

    Returns:
        brightest_indices: A list of the indices of the brightest component and its neighbors.
        brightest_*: Lists of the values of the ID, X, Y, RA, DEC, STON, FREQ, ALPHA, GAIN,
            FLAG, XX, YY, XY, YX, I, Q, U, and V of the brightest component and its neighbors.
    """

    # Finds the index of the brightest component in the EO.
    # If n > 0, then we search for the brightest of all components except the last n amount.
    if n == 0:
        brightest_index = EO_sources_I[i][j].index(max(EO_sources_I[i][j]))
    if n > 0:
        brightest_index = EO_sources_I[i][j].index(max(EO_sources_I[i][j]
                                                   [:-min(n, len(EO_sources_I[i][j]) - 1)]))


    # Finds the indices of the brightest component and all of its neighbors in a user-defined radius.
    # If n > 0, then we search for the brightest of all components and its neighbors except the last n amount.
    if n == 0:
        brightest_indices = [k for k, val in enumerate(EO_sources_I[i][j])
                             if np.sqrt((EO_sources_RA[i][j][k] - EO_sources_RA[i][j][brightest_index])**2 +
                             (EO_sources_DEC[i][j][k] - EO_sources_DEC[i][j][brightest_index])**2) < radius / 3600.]
    if n > 0:
        brightest_indices = [k for k, val in enumerate(EO_sources_I[i][j][:-min(n, len(EO_sources_I[i][j]) - 1)])
                             if np.sqrt((EO_sources_RA[i][j][k] - EO_sources_RA[i][j][brightest_index])**2 +
                             (EO_sources_DEC[i][j][k] - EO_sources_DEC[i][j][brightest_index])**2) < radius / 3600.]

    # Stores the ID, X, Y, RA, DEC, STON, FREQ, ALPHA, GAIN, FLAG,
    # XX, YY, XY, YX, I, Q, U, and V values of the brightest components and its neighbors
    # into individual lists.

    brightest_ID = [EO_sources_ID[i][j][val] for val in brightest_indices]
    brightest_X = [EO_sources_X[i][j][val] for val in brightest_indices]
    brightest_Y = [EO_sources_Y[i][j][val] for val in brightest_indices]
    brightest_RA = [EO_sources_RA[i][j][val] for val in brightest_indices]
    brightest_DEC = [EO_sources_DEC[i][j][val] for val in brightest_indices]
    brightest_STON = [EO_sources_STON[i][j][val] for val in brightest_indices]
    brightest_FREQ = [EO_sources_FREQ[i][j][val] for val in brightest_indices]
    brightest_ALPHA = [EO_sources_ALPHA[i][j][val] for val in brightest_indices]
    brightest_GAIN = [EO_sources_GAIN[i][j][val] for val in brightest_indices]
    brightest_FLAG = [EO_sources_FLAG[i][j][val] for val in brightest_indices]
    brightest_XX = [EO_sources_XX[i][j][val] for val in brightest_indices]
    brightest_YY = [EO_sources_YY[i][j][val] for val in brightest_indices]
    brightest_XY = [EO_sources_XY[i][j][val] for val in brightest_indices]
    brightest_YX = [EO_sources_YX[i][j][val] for val in brightest_indices]
    brightest_I = [EO_sources_I[i][j][val] for val in brightest_indices]
    brightest_Q = [EO_sources_Q[i][j][val] for val in brightest_indices]
    brightest_U = [EO_sources_U[i][j][val] for val in brightest_indices]
    brightest_V = [EO_sources_V[i][j][val] for val in brightest_indices]

    return {'brightest_indices': brightest_indices,
            'brightest_ID': brightest_ID,
            'brightest_X': brightest_X,
            'brightest_Y': brightest_Y,
            'brightest_RA': brightest_RA,
            'brightest_DEC': brightest_DEC,
            'brightest_STON': brightest_STON,
            'brightest_FREQ': brightest_FREQ,
            'brightest_ALPHA': brightest_ALPHA,
            'brightest_GAIN': brightest_GAIN,
            'brightest_FLAG': brightest_FLAG,
            'brightest_XX': brightest_XX,
            'brightest_YY': brightest_YY,
            'brightest_XY': brightest_XY,
            'brightest_YX': brightest_YX,
            'brightest_I': brightest_I,
            'brightest_Q': brightest_Q,
            'brightest_U': brightest_U,
            'brightest_V': brightest_V}


def beater(chased):
    """
    The Beater finds the weighted average of the position, STON, FREQ, ALPHA, and GAIN
    for an EO's brightest component and its neigbors as determined by the Chaser.
    
    The Beater also finds the sum of the flux for an EO's brightest component
    and its neigbors as determined by the Chaser.
    
    This is primarily written to be used by the Clusterer.
    
    Args:
        chased: The variable assigned to the brightest components as determined
            by the Chaser.

    Returns:
        Weighted and summed values for ID, X, Y, RA, DEC, STON, FREQ,
        ALPHA, GAIN, FLAG, XX, YY, XY, YX, I, Q, U, and V.
    """
    
    brightest_ID = chased['brightest_ID']
    brightest_X = chased['brightest_X']
    brightest_Y = chased['brightest_Y']
    brightest_RA = chased['brightest_RA']
    brightest_DEC = chased['brightest_DEC']
    brightest_STON = chased['brightest_STON']
    brightest_FREQ = chased['brightest_FREQ']
    brightest_ALPHA = chased['brightest_ALPHA']
    brightest_GAIN = chased['brightest_GAIN']
    brightest_FLAG = chased['brightest_FLAG']
    brightest_XX = chased['brightest_XX']
    brightest_YY = chased['brightest_YY']
    brightest_XY = chased['brightest_XY']
    brightest_YX = chased['brightest_YX']
    brightest_I = chased['brightest_I']
    brightest_Q = chased['brightest_Q']
    brightest_U = chased['brightest_U']
    brightest_V = chased['brightest_V']

    weighted_ID = np.average(brightest_ID, weights=brightest_I)
    weighted_X = np.average(brightest_X, weights=brightest_I)
    weighted_Y = np.average(brightest_Y, weights=brightest_I)
    weighted_RA = np.average(brightest_RA, weights=brightest_I)
    weighted_DEC = np.average(brightest_DEC, weights=brightest_I)
    weighted_STON = np.average(brightest_STON, weights=brightest_I)
    weighted_FREQ = np.average(brightest_FREQ, weights=brightest_I)
    weighted_ALPHA = np.average(brightest_ALPHA, weights=brightest_I)
    weighted_GAIN = np.average(brightest_GAIN, weights=brightest_I)
    weighted_FLAG = np.average(brightest_FLAG, weights=brightest_I)
    weighted_XX = sum(brightest_XX)
    weighted_YY = sum(brightest_YY)
    weighted_XY = sum(brightest_XY)
    weighted_YX = sum(brightest_YX)
    weighted_I = sum(brightest_I)
    weighted_Q = sum(brightest_Q)
    weighted_U = sum(brightest_U)
    weighted_V = sum(brightest_V)
    return {'weighted_ID': weighted_ID,
            'weighted_X': weighted_X,
            'weighted_Y': weighted_Y,
            'weighted_RA': weighted_RA,
            'weighted_DEC': weighted_DEC,
            'weighted_STON': weighted_STON,
            'weighted_FREQ': weighted_FREQ,
            'weighted_ALPHA': weighted_ALPHA,
            'weighted_GAIN': weighted_GAIN,
            'weighted_FLAG': weighted_FLAG,
            'weighted_XX': weighted_XX,
            'weighted_YY': weighted_YY,
            'weighted_XY': weighted_XY,
            'weighted_YX': weighted_YX,
            'weighted_I': weighted_I,
            'weighted_Q': weighted_Q,
            'weighted_U': weighted_U,
            'weighted_V': weighted_V}


def keeper(the_list, old_values, new_value):
    """
    The Keeper takes a list of extended source data, removes the brightest component
    and its neighbors (as determined by the Chaser), and appends a new, weighted
    component (as determined by the Beater).
    
    This is primarily written to be used by the Clusterer.
    
    Args:
        the_list: The list of EO data we are reformatting.
        old_values: The list of the brightest component and its neighbors (Chaser).
        new_value: The clustered component (Beater).

    Returns:
        A reformatted list with old components removed and a clustered component added.
    """

    # Making a copy of the list.
    reformatted_list = copy.copy(the_list)
    # Deleting the brightest component and its neighbors.
    # The order of these indices is reversed to avoid indexing errors as we remove data.
    for index in sorted(old_values, reverse=True):
        del reformatted_list[index]
    # Appending the clustered component.
    reformatted_list.append(new_value)
    return reformatted_list


def bright_keeper(bright_list, original_list, old_values):
    """
    The Bright Keeper keeps a record of the brightest component
    and its neighbors (as determined by the Chaser).
    
    This is primarily written to be used by the Clusterer.
    
    Args:
        bright_list: The list of bright, pre-clustered components we are reformatting.
        the_list: The original list of EO data as found by the seeker.
        old_values: The list of the brightest component and its neighbors (Chaser).

    Returns:
        A reformatted list containing bright, pre-clustered components.
    """

    # Making a copy of the list.
    reformatted_bright_list = copy.copy(bright_list)
    # Initializing the list as an empty list if the bright_list is the same as the original list.
    # This is necessary for the first iteration of clustering for each EO.
    if bright_list == original_list:
        reformatted_bright_list = []
    # Appending the pre-clustered components.
    [reformatted_bright_list.append([value for value in old_values])]
    return reformatted_bright_list


def weight_keeper(weight_list, original_list, new_value):
    """
    The Weight Keeper keeps a record of the clustered components (as determined 
    by the Beater).
    
    This is primarily written to be used by the Clusterer.
    
    Args:
        weight_list: The list of clustered components we are reformatting.
        the_list: The original list of EO data as found by the Seeker.
        new_value: The clustered component (Beater).

    Returns:
        A reformatted list containing bright, pre-clustered components.
    """

    # Making a copy of the list.
    reformatted_weight_list = copy.copy(weight_list)
    # Initializing the list as an empty list if the bright_list is the same as the original list.
    # This is necessary for the first iteration of clustering for each EO.
    if weight_list == original_list:
        reformatted_weight_list = []
    # Appending the clustered component.
    reformatted_weight_list.append(new_value)
    return reformatted_weight_list


def clusterer(data, radius=10, cutoff=.03):
    """
    The Clusterer takes data which has been read in by the collector.
    It iteratively searches for the brightest components of extended objects,
    and clusters data points within a user-defined radius in arcseconds. It then
    subtracts all clustered sources which do not pass a user-defined flux cutoff in Janskies.
    
    Args:
        data: The variable assigned to the IDL .sav file which has been
            read in by the collector.
        radius: The radius in arcseconds for components to be clustered.
        cutoff: The cutoff in Janskies for components to be removed.

    Returns:
        EO_sources_*: Lists of the ID, X, Y, RA, DEC, STON, FREQ, ALPHA, GAIN,
            FLAG, XX, YY, XY, YX, I, Q, U, and V of clustered extended objects.
        added_EO_sources_*: Lists of the RA, DEC, and I of clustered components.
        removed_EO_sources_*: Lists of the RA, DEC, and I of pre-clustered components.
    """

    # The Seeker reads in the data file.
    # Next, we initialize the data we will be iterating through as we cluster.
    sought = seeker(data)
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
    added_EO_sources_ID = copy.deepcopy(sought['EO_sources_ID'])
    added_EO_sources_X = copy.deepcopy(sought['EO_sources_X'])
    added_EO_sources_Y = copy.deepcopy(sought['EO_sources_Y'])
    added_EO_sources_RA = copy.deepcopy(sought['EO_sources_RA'])
    added_EO_sources_DEC = copy.deepcopy(sought['EO_sources_DEC'])
    added_EO_sources_STON = copy.deepcopy(sought['EO_sources_STON'])
    added_EO_sources_FREQ = copy.deepcopy(sought['EO_sources_FREQ'])
    added_EO_sources_ALPHA = copy.deepcopy(sought['EO_sources_ALPHA'])
    added_EO_sources_GAIN = copy.deepcopy(sought['EO_sources_GAIN'])
    added_EO_sources_FLAG = copy.deepcopy(sought['EO_sources_FLAG'])
    added_EO_sources_XX = copy.deepcopy(sought['EO_sources_XX'])
    added_EO_sources_YY = copy.deepcopy(sought['EO_sources_YY'])
    added_EO_sources_XY = copy.deepcopy(sought['EO_sources_XY'])
    added_EO_sources_YX = copy.deepcopy(sought['EO_sources_YX'])
    added_EO_sources_I = copy.deepcopy(sought['EO_sources_I'])
    added_EO_sources_Q = copy.deepcopy(sought['EO_sources_Q'])
    added_EO_sources_U = copy.deepcopy(sought['EO_sources_U'])
    added_EO_sources_V = copy.deepcopy(sought['EO_sources_V'])
    removed_EO_sources_ID = copy.deepcopy(sought['EO_sources_ID'])
    removed_EO_sources_X = copy.deepcopy(sought['EO_sources_X'])
    removed_EO_sources_Y = copy.deepcopy(sought['EO_sources_Y'])
    removed_EO_sources_RA = copy.deepcopy(sought['EO_sources_RA'])
    removed_EO_sources_DEC = copy.deepcopy(sought['EO_sources_DEC'])
    removed_EO_sources_STON = copy.deepcopy(sought['EO_sources_STON'])
    removed_EO_sources_FREQ = copy.deepcopy(sought['EO_sources_FREQ'])
    removed_EO_sources_ALPHA = copy.deepcopy(sought['EO_sources_ALPHA'])
    removed_EO_sources_GAIN = copy.deepcopy(sought['EO_sources_GAIN'])
    removed_EO_sources_FLAG = copy.deepcopy(sought['EO_sources_FLAG'])
    removed_EO_sources_XX = copy.deepcopy(sought['EO_sources_XX'])
    removed_EO_sources_YY = copy.deepcopy(sought['EO_sources_YY'])
    removed_EO_sources_XY = copy.deepcopy(sought['EO_sources_XY'])
    removed_EO_sources_YX = copy.deepcopy(sought['EO_sources_YX'])
    removed_EO_sources_I = copy.deepcopy(sought['EO_sources_I'])
    removed_EO_sources_Q = copy.deepcopy(sought['EO_sources_Q'])
    removed_EO_sources_U = copy.deepcopy(sought['EO_sources_U'])
    removed_EO_sources_V = copy.deepcopy(sought['EO_sources_V'])

    for i in range(len(data['data'])):
        for j in range(len(EO_sources_I[i])):
            print "Currently clustering i:", i, "j:", j

            # Initializing the n value we feed into the Chaser.
            # This n value will be useful in restricting the Clusterer from
            # redundantly choosing a the same component it has already clustered
            # while it loops to find next brightest components.

            m = 0

            # We will now loop through each extended object until all components have been clustered.

            while m < len(EO_sources_RA[i][j]):
                # The Chaser finds the values and indices of the brightest
                # non-clustered component and its neighbors within a user-defined radius.
                chased = chaser(EO_sources_ID, EO_sources_X, EO_sources_Y,
                                EO_sources_RA, EO_sources_DEC, EO_sources_STON,
                                EO_sources_FREQ, EO_sources_ALPHA, EO_sources_GAIN,
                                EO_sources_FLAG, EO_sources_XX, EO_sources_YY,
                                EO_sources_XY, EO_sources_YX, EO_sources_I,
                                EO_sources_Q, EO_sources_U, EO_sources_V,
                                radius, i=i, j=j, n=m)
                brightest_indices = chased['brightest_indices']
                brightest_ID = chased['brightest_ID']
                brightest_X = chased['brightest_X']
                brightest_Y = chased['brightest_Y']
                brightest_RA = chased['brightest_RA']
                brightest_DEC = chased['brightest_DEC']
                brightest_STON = chased['brightest_STON']
                brightest_FREQ = chased['brightest_FREQ']
                brightest_ALPHA = chased['brightest_ALPHA']
                brightest_GAIN = chased['brightest_GAIN']
                brightest_FLAG = chased['brightest_FLAG']
                brightest_XX = chased['brightest_XX']
                brightest_YY = chased['brightest_YY']
                brightest_XY = chased['brightest_XY']
                brightest_YX = chased['brightest_YX']
                brightest_I = chased['brightest_I']
                brightest_Q = chased['brightest_Q']
                brightest_U = chased['brightest_U']
                brightest_V = chased['brightest_V']

                # The Beater takes the components read in by the chaser and clusters them.
                weighted = beater(chased)
                weighted_ID = weighted['weighted_ID']
                weighted_X = weighted['weighted_X']
                weighted_Y = weighted['weighted_Y']
                weighted_RA = weighted['weighted_RA']
                weighted_DEC = weighted['weighted_DEC']
                weighted_STON = weighted['weighted_STON']
                weighted_FREQ = weighted['weighted_FREQ']
                weighted_ALPHA = weighted['weighted_ALPHA']
                weighted_GAIN = weighted['weighted_GAIN']
                weighted_FLAG = weighted['weighted_FLAG']
                weighted_XX = weighted['weighted_XX']
                weighted_YY = weighted['weighted_YY']
                weighted_XY = weighted['weighted_XY']
                weighted_YX = weighted['weighted_YX']
                weighted_I = weighted['weighted_I']
                weighted_Q = weighted['weighted_Q']
                weighted_U = weighted['weighted_U']
                weighted_V = weighted['weighted_V']

                # The Keeper reformats the EO components lists and replaces pre-clustered
                # components determined by the Chaser with the clustered component determined
                # by the Beater.

                EO_sources_ID[i][j] = copy.deepcopy(keeper(EO_sources_ID[i][j],
                                                    brightest_indices, weighted_ID))
                EO_sources_X[i][j] = copy.deepcopy(keeper(EO_sources_X[i][j],
                                                   brightest_indices, weighted_X))
                EO_sources_Y[i][j] = copy.deepcopy(keeper(EO_sources_Y[i][j],
                                                   brightest_indices, weighted_Y))
                EO_sources_RA[i][j] = copy.deepcopy(keeper(EO_sources_RA[i][j],
                                                    brightest_indices, weighted_RA))
                EO_sources_DEC[i][j] = copy.deepcopy(keeper(EO_sources_DEC[i][j],
                                                     brightest_indices, weighted_DEC))
                EO_sources_STON[i][j] = copy.deepcopy(keeper(EO_sources_STON[i][j],
                                                      brightest_indices, weighted_STON))
                EO_sources_FREQ[i][j] = copy.deepcopy(keeper(EO_sources_FREQ[i][j],
                                                      brightest_indices, weighted_FREQ))
                EO_sources_ALPHA[i][j] = copy.deepcopy(keeper(EO_sources_ALPHA[i][j],
                                                       brightest_indices, weighted_ALPHA))
                EO_sources_GAIN[i][j] = copy.deepcopy(keeper(EO_sources_GAIN[i][j],
                                                      brightest_indices, weighted_GAIN))
                EO_sources_FLAG[i][j] = copy.deepcopy(keeper(EO_sources_FLAG[i][j],
                                                      brightest_indices, weighted_FLAG))
                EO_sources_XX[i][j] = copy.deepcopy(keeper(EO_sources_XX[i][j],
                                                    brightest_indices, weighted_XX))
                EO_sources_YY[i][j] = copy.deepcopy(keeper(EO_sources_YY[i][j],
                                                    brightest_indices, weighted_YY))
                EO_sources_XY[i][j] = copy.deepcopy(keeper(EO_sources_XY[i][j],
                                                    brightest_indices, weighted_XY))
                EO_sources_YX[i][j] = copy.deepcopy(keeper(EO_sources_YX[i][j],
                                                    brightest_indices, weighted_YX))
                EO_sources_I[i][j] = copy.deepcopy(keeper(EO_sources_I[i][j],
                                                   brightest_indices, weighted_I))
                EO_sources_Q[i][j] = copy.deepcopy(keeper(EO_sources_Q[i][j],
                                                   brightest_indices, weighted_Q))
                EO_sources_U[i][j] = copy.deepcopy(keeper(EO_sources_U[i][j],
                                                   brightest_indices, weighted_U))
                EO_sources_V[i][j] = copy.deepcopy(keeper(EO_sources_V[i][j],
                                                   brightest_indices, weighted_V))
                added_EO_sources_RA[i][j] = copy.deepcopy(weight_keeper(added_EO_sources_RA[i][j],
                                                          sought['EO_sources_RA'][i][j], weighted_RA))
                added_EO_sources_DEC[i][j] = copy.deepcopy(weight_keeper(added_EO_sources_DEC[i][j],
                                                           sought['EO_sources_DEC'][i][j], weighted_DEC))
                added_EO_sources_I[i][j] = copy.deepcopy(weight_keeper(added_EO_sources_I[i][j],
                                                         sought['EO_sources_I'][i][j], weighted_I))
                removed_EO_sources_RA[i][j] = copy.deepcopy(bright_keeper(removed_EO_sources_RA[i][j],
                                                            sought['EO_sources_RA'][i][j], brightest_RA))
                removed_EO_sources_DEC[i][j] = copy.deepcopy(bright_keeper(removed_EO_sources_DEC[i][j],
                                                             sought['EO_sources_DEC'][i][j], brightest_DEC))
                removed_EO_sources_I[i][j] = copy.deepcopy(bright_keeper(removed_EO_sources_I[i][j],
                                                           sought['EO_sources_I'][i][j], brightest_I))
                m += 1

    for i in range(len(data['data'])):
        for j in range(len(EO_sources_I[i])):
            # We now find which clustered components do not meet a minimum, user-defined
            # flux threshold in Janskies and subsequently remove them from all lists.
            faintremover_indices = [k for k, kval in enumerate(EO_sources_I[i][j])
                                    if kval < cutoff]
            for index in sorted(faintremover_indices, reverse=True):
                del EO_sources_ID[i][j][index]
                del EO_sources_X[i][j][index]
                del EO_sources_Y[i][j][index]
                del EO_sources_RA[i][j][index]
                del EO_sources_DEC[i][j][index]
                del EO_sources_STON[i][j][index]
                del EO_sources_FREQ[i][j][index]
                del EO_sources_ALPHA[i][j][index]
                del EO_sources_GAIN[i][j][index]
                del EO_sources_FLAG[i][j][index]
                del EO_sources_XX[i][j][index]
                del EO_sources_YY[i][j][index]
                del EO_sources_XY[i][j][index]
                del EO_sources_YX[i][j][index]
                del EO_sources_I[i][j][index]
                del EO_sources_Q[i][j][index]
                del EO_sources_U[i][j][index]
                del EO_sources_V[i][j][index]
                del added_EO_sources_RA[i][j][index]
                del added_EO_sources_DEC[i][j][index]
                del added_EO_sources_I[i][j][index]
                del removed_EO_sources_RA[i][j][index]
                del removed_EO_sources_DEC[i][j][index]
                del removed_EO_sources_I[i][j][index]

    return {'EO_sources_ID': EO_sources_ID,
            'EO_sources_X': EO_sources_X,
            'EO_sources_Y': EO_sources_Y,
            'EO_sources_RA': EO_sources_RA,
            'EO_sources_DEC': EO_sources_DEC,
            'EO_sources_STON': EO_sources_STON,
            'EO_sources_FREQ': EO_sources_FREQ,
            'EO_sources_ALPHA': EO_sources_ALPHA,
            'EO_sources_GAIN': EO_sources_GAIN,
            'EO_sources_FLAG': EO_sources_FLAG,
            'EO_sources_XX': EO_sources_XX,
            'EO_sources_YY': EO_sources_YY,
            'EO_sources_XY': EO_sources_XY,
            'EO_sources_YX': EO_sources_YX,
            'EO_sources_I': EO_sources_I,
            'EO_sources_Q': EO_sources_Q,
            'EO_sources_U': EO_sources_U,
            'EO_sources_V': EO_sources_V,
            'added_EO_sources_RA': added_EO_sources_RA,
            'added_EO_sources_DEC': added_EO_sources_DEC,
            'added_EO_sources_I': added_EO_sources_I,
            'removed_EO_sources_RA': removed_EO_sources_RA,
            'removed_EO_sources_DEC': removed_EO_sources_DEC,
            'removed_EO_sources_I': removed_EO_sources_I}


def framer(RA, DEC):
    """
    The Framer gives RA and DEC values to center EOs for plotting.
    
    Args:
        RA: List of all RA values for an extended object.
        DEC: List of all DEC values for an extended object.

    Returns:
        Minimum and maximum values of RA and DEC to center an EO for plotting.
    """
    min_RA, max_RA, min_DEC, max_DEC = min(RA), max(RA), min(DEC), max(DEC)
    min_degrees = 25
    x = max_RA - min_RA
    y = max_DEC - min_DEC
    z = .05 * max(x, y)
    if (x > y) and (x - y > min_degrees / 3600.):
        RA_zoom_min = min_RA - z
        RA_zoom_max = max_RA + z
        DEC_zoom_min = min_DEC - .5 * (x - y) - z
        DEC_zoom_max = max_DEC + .5 * (x - y) + z
    elif (y > x) and (y - x > min_degrees / 3600.):
        RA_zoom_min = min_RA - .5 * (y - x) - z
        RA_zoom_max = max_RA + .5 * (y - x) + z
        DEC_zoom_min = min_DEC - z
        DEC_zoom_max = max_DEC + z
    else:
        RA_zoom_min = RA[0] - .25
        RA_zoom_max = RA[-1] + .25
        DEC_zoom_min = DEC[0] - .25
        DEC_zoom_max = DEC[-1] + .25
    return {'RA_zoom_min': RA_zoom_min, 'RA_zoom_max': RA_zoom_max,
            'DEC_zoom_min': DEC_zoom_min, 'DEC_zoom_max': DEC_zoom_max}


def modelcomparer(data, radius=10, cutoff=.02, binwidth=2,
                  sigmawidth=4, minI=50, minBeam=.75, EOid=None,
                  plotdata=False, saveall=True, savebefore=True,
                  savebeforeafter=True):

    separated = separator(data)

    modeled = modeler(data, radius, cutoff)
    modeled_RA = copy.deepcopy(modeled['EO_sources_RA'])
    modeled_DEC = copy.deepcopy(modeled['EO_sources_DEC'])
    modeled_I = copy.deepcopy(modeled['EO_sources_I'])
    added_RA = copy.deepcopy(modeled['added_EO_sources_RA'])
    added_DEC = copy.deepcopy(modeled['added_EO_sources_DEC'])
    added_I = copy.deepcopy(modeled['added_EO_sources_I'])
    removed_RA = copy.deepcopy(modeled['removed_EO_sources_RA'])
    removed_DEC = copy.deepcopy(modeled['removed_EO_sources_DEC'])
    removed_I = copy.deepcopy(modeled['removed_EO_sources_I'])

    sought = seeker(data)
    unmodeled_RA = copy.deepcopy(sought['EO_sources_RA'])
    unmodeled_DEC = copy.deepcopy(sought['EO_sources_DEC'])
    unmodeled_I = copy.deepcopy(sought['EO_sources_I'])
    unmodeled_XX = copy.deepcopy(sought['EO_sources_XX'])
    unmodeled_YY = copy.deepcopy(sought['EO_sources_YY'])
    all_RA = copy.deepcopy(sought['all_RA'])
    all_DEC = copy.deepcopy(sought['all_DEC'])
    all_I = copy.deepcopy(sought['all_I'])
    all_XX = copy.deepcopy(sought['all_XX'])
    all_YY = copy.deepcopy(sought['all_YY'])
    all_BEAM = copy.deepcopy(sought['all_BEAM'])

    pixelreplacement = 1e-5
    cmap = matplotlib.cm.get_cmap('gist_heat')
    sigmawidth = sigmawidth * (1. / 3600.)
    binwidth = binwidth * (1. / 3600.)
    kernel = Gaussian2DKernel(stddev=(sigmawidth / binwidth))

    for i in range(len(modeled_RA)):
        for j in range(len(modeled_RA[i])):
            if EOid is not None:
                minI, minBeam = 0, 0
            if (EOid is not None) and (separated['extsources'][i][j]['ID'] != EOid):
                continue
            if (sum(unmodeled_I[i][j]) > minI) and ((sum(unmodeled_XX[i][j]) + sum(unmodeled_YY[i][j])) / sum(unmodeled_I[i][j]) > minBeam):
                print "Currently plotting EOid:", separated['extsources'][i][j]['ID']
                if separated['extsources'][i][j]['RA'] > 180:
                    separated['extsources'][i][j]['RA'] -= 360
                diffuse_RA_total = copy.deepcopy(unmodeled_RA[i][j])
                diffuse_DEC_total = copy.deepcopy(unmodeled_DEC[i][j])
                diffuse_I_total = copy.deepcopy(unmodeled_I[i][j])

                all_framed = framer(all_RA[i], all_DEC[i])
                all_RA_zoom = [all_framed['RA_zoom_min'], all_framed['RA_zoom_max']]
                all_DEC_zoom = [all_framed['DEC_zoom_min'], all_framed['DEC_zoom_max']]
                all_n_bins = min(int(max(all_RA_zoom[1] - all_RA_zoom[0], all_DEC_zoom[1] - all_DEC_zoom[0]) / binwidth), 1024)
                all_RA_total = np.array(all_RA[i])
                all_DEC_total = np.array(all_DEC[i])
                all_I_total = np.array(all_I[i])
                all_XX_total = np.array(all_XX[i])
                all_YY_total = np.array(all_YY[i])
                all_BEAM_total = np.array(all_BEAM[i])
                (all_pixels, all_RA_pixel_centers, all_DEC_pixel_centers) = pixelate(
                    all_RA_zoom, all_DEC_zoom, all_n_bins, all_RA_total, all_DEC_total, all_I_total)
                (all_pixels_contour, all_RA_pixel_centers_contour, all_DEC_pixel_centers_contour) = pixelate(
                    all_RA_zoom, all_DEC_zoom, 50, all_RA_total, all_DEC_total, all_BEAM_total)
                all_pixels[all_pixels == 0] = pixelreplacement
                all_convolved = convolve_fft(all_pixels, kernel)

                unmodeled_framed = framer(unmodeled_RA[i][j], unmodeled_DEC[i][j])
                unmodeled_RA_zoom = [unmodeled_framed['RA_zoom_min'], unmodeled_framed['RA_zoom_max']]
                unmodeled_DEC_zoom = [unmodeled_framed['DEC_zoom_min'], unmodeled_framed['DEC_zoom_max']]
                unmodeled_RA_total = np.array(unmodeled_RA[i][j])
                unmodeled_DEC_total = np.array(unmodeled_DEC[i][j])
                unmodeled_I_total = np.array(unmodeled_I[i][j])
                unmodeled_n_bins = int((unmodeled_framed['RA_zoom_max'] - unmodeled_framed['RA_zoom_min']) / binwidth)
                (unmodeled_pixels, unmodeled_RA_pixel_centers, unmodeled_DEC_pixel_centers) = \
                    pixelate(unmodeled_RA_zoom, unmodeled_DEC_zoom, unmodeled_n_bins,
                             unmodeled_RA_total, unmodeled_DEC_total, unmodeled_I_total)
                unmodeled_pixels[unmodeled_pixels == 0] = pixelreplacement
                unmodeled_pixels = convolve(unmodeled_pixels, kernel)

                modeled_RA_zoom = unmodeled_RA_zoom
                modeled_DEC_zoom = unmodeled_DEC_zoom
                modeled_RA_total = np.array(modeled_RA[i][j])
                modeled_DEC_total = np.array(modeled_DEC[i][j])
                modeled_I_total = np.array(modeled_I[i][j])
                modeled_n_bins = unmodeled_n_bins
                (modeled_pixels, modeled_RA_pixel_centers, modeled_DEC_pixel_centers) = \
                    pixelate(modeled_RA_zoom, modeled_DEC_zoom, modeled_n_bins,
                             modeled_RA_total, modeled_DEC_total, modeled_I_total)
                modeled_pixels[modeled_pixels == 0] = pixelreplacement
                modeled_pixels = convolve(modeled_pixels, kernel)

                added_RA_zoom = unmodeled_RA_zoom
                added_DEC_zoom = unmodeled_DEC_zoom
                added_RA_total = np.array(added_RA[i][j])
                added_DEC_total = np.array(added_DEC[i][j])
                added_I_total = np.array(added_I[i][j])
                added_n_bins = unmodeled_n_bins
                (added_pixels, added_RA_pixel_centers, added_DEC_pixel_centers) = \
                    pixelate(added_RA_zoom, added_DEC_zoom, added_n_bins,
                             added_RA_total, added_DEC_total, added_I_total)
                added_pixels = convolve(added_pixels, kernel)

                removed_RA_zoom = unmodeled_RA_zoom
                removed_DEC_zoom = unmodeled_DEC_zoom
                removed_RA_total = np.array([removed_RA[i][j][k][z]
                                             for k in range(len(removed_RA[i][j]))
                                             for z in range(len(removed_RA[i][j][k]))])
                removed_DEC_total = np.array([removed_DEC[i][j][k][z]
                                              for k in range(len(removed_DEC[i][j]))
                                              for z in range(len(removed_DEC[i][j][k]))])
                removed_I_total = np.array([removed_I[i][j][k][z]
                                            for k in range(len(removed_I[i][j]))
                                            for z in range(len(removed_I[i][j][k]))])
                removed_n_bins = unmodeled_n_bins
                (removed_pixels, removed_RA_pixel_centers, removed_DEC_pixel_centers) = \
                    pixelate(removed_RA_zoom, removed_DEC_zoom, removed_n_bins,
                             removed_RA_total, removed_DEC_total, removed_I_total)
                removed_pixels[removed_pixels == 0] = pixelreplacement
                removed_pixels = convolve(removed_pixels, kernel)

                diffuse_RA_zoom = unmodeled_RA_zoom
                diffuse_DEC_zoom = unmodeled_DEC_zoom
                [diffuse_RA_total.remove(val) for val in removed_RA_total if val in diffuse_RA_total]
                diffuse_RA_total = np.array(diffuse_RA_total)
                [diffuse_DEC_total.remove(val) for val in removed_DEC_total if val in diffuse_DEC_total]
                diffuse_DEC_total = np.array(diffuse_DEC_total)
                [diffuse_I_total.remove(val) for val in removed_I_total if val in diffuse_I_total]
                diffuse_I_total = np.array(diffuse_I_total)
                diffuse_n_bins = unmodeled_n_bins
                (diffuse_pixels, diffuse_RA_pixel_centers, diffuse_DEC_pixel_centers) = \
                    pixelate(diffuse_RA_zoom, diffuse_DEC_zoom, diffuse_n_bins,
                             diffuse_RA_total, diffuse_DEC_total, diffuse_I_total)
                diffuse_pixels[diffuse_pixels == 0] = pixelreplacement
                diffuse_pixels = convolve(diffuse_pixels, kernel)

                ###
                height_ratios = [1, 4, .5, 4]
                width_ratios = [4, .25, 1.5, 4, .25, 1.5, 4, .25]
                wspace = 0.05
                hspace = 0.05
                fig = plt.figure(figsize=(sum(width_ratios) + wspace * (len(width_ratios) - 1),
                                          sum(height_ratios) + hspace * (len(height_ratios) - 1)))
                gs = gridspec.GridSpec(len(height_ratios), len(width_ratios), height_ratios=height_ratios, width_ratios=width_ratios)
                gs.update(left=0, right=1, bottom=0,
                          top=1, wspace=wspace, hspace=hspace)

                bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="red", lw=2)

                ###

                fig.suptitle('Clustering of EO Source {} from ObsID {}'.format
                             (separated['extsources'][i][j]['ID'],
                              [int(s) for s in re.findall('\d+', data['filenames'][i])][0],
                              separated['extsources'][i][j]['FREQ']), fontsize=20)

                # Unmodeled Plot
                ax1 = fig.add_subplot(gs[1, 0])
                unmodeled = ax1.imshow(np.transpose(unmodeled_pixels), norm=LogNorm(vmin=pixelreplacement, vmax=1),
                                       origin="lower", interpolation="none", cmap=cmap,
                                       extent=[unmodeled_RA_pixel_centers[0], unmodeled_RA_pixel_centers[-1],
                                               unmodeled_DEC_pixel_centers[0], unmodeled_DEC_pixel_centers[-1]])
                ax1.update({'title': 'Pre-clustered with {} components, {:.4G} Jys'
                            .format(len(unmodeled_RA[i][j]), sum(unmodeled_I[i][j])),
                            'xlabel': 'RA',
                            'ylabel': 'DEC'})
                ax1.grid(True, ls='dashed', alpha=.75)
                ax1.set_xlim(ax1.get_xlim()[::-1])

                # Modeled Plot
                ax2 = fig.add_subplot(gs[1, 3])
                modeled = ax2.imshow(np.transpose(modeled_pixels), norm=LogNorm(vmin=pixelreplacement, vmax=1),
                                     origin="lower", interpolation="none", cmap=cmap,
                                     extent=[modeled_RA_pixel_centers[0], modeled_RA_pixel_centers[-1],
                                             modeled_DEC_pixel_centers[0], modeled_DEC_pixel_centers[-1]])
                ax2.update({'title': 'Clustered with {} components, {:.4G} Jys'
                            .format(len(modeled_RA[i][j]), sum(modeled_I[i][j])),
                            'xlabel': 'RA',
                            'ylabel': 'DEC'})
                if max(removed_RA_total) == 0:
                    no_modeled = ax2.text(np.mean(modeled_RA_pixel_centers),
                                          np.mean(modeled_DEC_pixel_centers),
                                          "No Modeling Occurred", ha="center", va="center",
                                          size=15, bbox=bbox_props)
                    ax2.update({'title': ''})
                ax2.grid(True, ls='dashed', alpha=.75)
                ax2.set_xlim(ax2.get_xlim()[::-1])

                # Modeled Pre-compacted Scatter
                ax3 = fig.add_subplot(gs[1, 6])
                removed = ax3.scatter(removed_RA_total, removed_DEC_total,
                                      c=np.linspace(1, 0, len(removed_I_total)),
                                      cmap=plt.cm.summer, s=2)
                rectangle_length = max((max(removed_RA_total) - min(removed_RA_total)),
                                       (max(removed_DEC_total) - min(removed_DEC_total)))
                ax3.add_patch(patches.Rectangle((min(removed_RA_total) - .1 * rectangle_length,
                                                 min(removed_DEC_total) - .1 * rectangle_length),
                                                1.2 * rectangle_length, 1.2 * rectangle_length,
                                                fill=False, color='coral'))
                ax3.grid(True, ls='dashed', alpha=.75)
                ax3.update({'title': 'Pre-compacted with {} components'.format(len(removed_RA[i][j])),
                            'xlabel': 'RA',
                            'ylabel': 'DEC',
                            'xlim': modeled_RA_zoom,
                            'ylim': modeled_DEC_zoom})
                if max(removed_RA_total) == 0:
                    no_modeled = ax3.text(np.mean(modeled_RA_pixel_centers),
                                          np.mean(modeled_DEC_pixel_centers),
                                          "No Modeling Occurred", ha="center", va="center",
                                          size=15, bbox=bbox_props)
                    ax3.update({'title': ''})
                ax3.set_xlim(ax3.get_xlim()[::-1])
                # Modeled Uncompacted
                ax4 = fig.add_subplot(gs[3, 0])
                all_plot = ax4.imshow(np.transpose(all_convolved), norm=LogNorm(vmin=pixelreplacement),
                                      origin="lower", interpolation="none", cmap=cmap,
                                      extent=[all_RA_pixel_centers[0], all_RA_pixel_centers[-1],
                                              all_DEC_pixel_centers[0], all_DEC_pixel_centers[-1]])
                contour_plot = ax4.contourf(np.transpose(all_pixels_contour), cmap=plt.cm.bone, alpha=.4,
                                            levels=np.linspace(min(all_BEAM_total), max(all_BEAM_total), 6),
                                            extent=[all_RA_pixel_centers[0], all_RA_pixel_centers[-1],
                                                    all_DEC_pixel_centers[0], all_DEC_pixel_centers[-1]])
                location = ax4.scatter(separated['extsources'][i][j]['RA'], separated['extsources'][i][j]['DEC'],
                                       color="cyan", s=60)
                ax4.clabel(contour_plot, fontsize=8,
                           inline=False, colors='white', alpha=1)
                ax4.update({'title': 'Observation',
                            'xlabel': 'RA',
                            'ylabel': 'DEC'})
                ax4.grid(True, ls='dashed', alpha=.75)
                ax4.set_xlim(ax4.get_xlim()[::-1])
                # Modeled Compacted
                ax5 = fig.add_subplot(gs[3, 3])
                diffuse = ax5.imshow(np.transpose(diffuse_pixels), norm=LogNorm(vmin=pixelreplacement, vmax=1),
                                     origin="lower", interpolation="none", cmap=plt.cm.bone,
                                     extent=[unmodeled_RA_pixel_centers[0], unmodeled_RA_pixel_centers[-1],
                                             unmodeled_DEC_pixel_centers[0], unmodeled_DEC_pixel_centers[-1]])
                added = ax5.imshow(np.transpose(added_pixels), norm=LogNorm(vmin=pixelreplacement, vmax=1),
                                   origin="lower", interpolation="none", cmap=cmap,
                                   extent=[modeled_RA_pixel_centers[0], modeled_RA_pixel_centers[-1],
                                           modeled_DEC_pixel_centers[0], modeled_DEC_pixel_centers[-1]])
                ax5.update({'title': 'Compacted with {} componenets, {:.4G} Janskies'.format(len(added_RA[i][j]), sum(added_I[i][j])),
                            'xlabel': 'RA',
                            'ylabel': 'DEC'})
                if max(removed_RA_total) == 0:
                    no_modeled = ax5.text(np.mean(modeled_RA_pixel_centers),
                                          np.mean(modeled_DEC_pixel_centers),
                                          "No Modeling Occurred", ha="center", va="center",
                                          size=15, bbox=bbox_props)
                    ax5.update({'title': ''})
                ax5.grid(True, ls='dashed', alpha=.75)
                ax5.set_xlim(ax5.get_xlim()[::-1])

                # Modeled Pre-compacted Zoom
                ax6 = fig.add_subplot(gs[3, 6])
                removed_zoom = ax6.scatter(removed_RA_total, removed_DEC_total,
                                           c=np.linspace(1, 0, len(removed_I_total)),
                                           cmap=plt.cm.summer, s=20)
                removed_centers = ax6.scatter(added_RA_total, added_DEC_total, c='red', marker='2', s=50)
                ax6.update({'title': 'Pre-compacted Components [Zoomed]',
                            'xlabel': 'RA',
                            'ylabel': 'DEC',
                            'xlim': [min(removed_RA_total) - .1 * rectangle_length,
                                     min(removed_RA_total) + 1.2 * rectangle_length],
                            'ylim': [min(removed_DEC_total) - .1 * rectangle_length,
                                     min(removed_DEC_total) + 1.2 * rectangle_length]})
                if max(removed_RA_total) == 0:
                    no_modeled = ax6.text(np.mean(removed_RA_total),
                                          np.mean(removed_DEC_total),
                                          "No Modeling Occurred", ha="center", va="center",
                                          size=15, bbox=bbox_props)
                    ax6.update({'title': ''})
                ax6.grid(True, ls='dashed', alpha=.75)
                ax6.set_xlim(ax6.get_xlim()[::-1])
                # Colorbars

                unmodeled_cbax = plt.subplot(gs[1, 1])
                unmodeled_cb = Colorbar(ax=unmodeled_cbax, mappable=unmodeled, orientation='vertical', ticklocation='right')
                unmodeled_cb.set_label(r'Janskies', labelpad=10)

                modeled_cbax = plt.subplot(gs[1, 4])
                modeled_cb = Colorbar(ax=modeled_cbax, mappable=modeled, orientation='vertical', ticklocation='right')
                modeled_cb.set_label(r'Janskies', labelpad=10)

                removed_cbax = plt.subplot(gs[1, 7])
                removed_cb = Colorbar(ax=removed_cbax, mappable=removed, orientation='vertical', ticklocation='right')
                removed_cb.set_ticks([1, 0])
                removed_cb.set_ticklabels(['First', 'Last'])

                removed_zoom_cbax = plt.subplot(gs[3, 7])
                removed_zoom_cb = Colorbar(ax=removed_zoom_cbax, mappable=removed_zoom, orientation='vertical', ticklocation='right')
                removed_zoom_cb.set_ticks([1, 0])
                removed_zoom_cb.set_ticklabels(['First', 'Last'])
                if saveall is True:
                    plt.savefig('EO' + '{}radius{}cutoff{}'.format(separated['extsources'][i][j]['ID'], radius, cutoff) + '.png', bbox_inches='tight')
                if savebefore is True:
                    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig('EO' + '{}radius{}cutoff{}'.format(separated['extsources'][i][j]['ID'], radius, cutoff) + 'before.png', bbox_inches=extent)
                if savebeforeafter is True:
                    extent = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    plt.savefig('EO' + '{}radius{}cutoff{}'.format(separated['extsources'][i][j]['ID'], radius, cutoff) + 'beforeafter.png', bbox_inches=extent)
                if plotdata is True:
                    plt.show()
