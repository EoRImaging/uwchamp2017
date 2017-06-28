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

    point_data = [data['data'][i]['source_array'][j] \
    if data['data'][i]['source_array'][j][-2] is None else ''\
    for i in range(len(data['data'])) \
    for j in range(len(data['data'][i]['source_array'])) ]
    
    extended_data = [data['data'][i]['source_array'][j] \
    if data['data'][i]['source_array'][j][-2] is not None else ''\
    for i in range(len(data['data'])) \
    for j in range(len(data['data'][i]['source_array'])) ]

    return {'extsources':extended_data,'psources':point_data}

def plotext(data,minI):
    
    extended_data = separator(data)[0]
    indi_I = []
    indi_I_nest = []
    indi_I_nest2 = []
    
    for i in range(len(data[0])):
        for j in range(len(extended_data[i][0]['EXTEND'])):
            for k in range(len(extended_data[i][j]['EXTEND']['FLUX'])):
                indi_I_nest2.append(extended_data[i][j]['EXTEND']['FLUX'][k]['I'][0])
            indi_I_nest.append(indi_I_nest2)
            indi_I_nest2 = []
        indi_I.append(indi_I_nest)
        indi_I_nest = []

    for i in range(len(data[0])):
        for j in range(len(extended_data[i][j]['EXTEND'])):
            if max(indi_I[i][j]) > minI:
                plt.figure()
                plt.scatter(extended_data[i][j]['EXTEND']['RA'],extended_data[i][j]['EXTEND']['DEC'],s=50,c=indi_I[i][j])
                plt.colorbar(label='Janskies')
                plt.figtext(0, -.15, "ObsID: {}\nTotal Flux: {}\nMean RA: {}\nMean Dec: {}".format(data[1][i],np.sum(indi_I[i][j]),extended_data[i][j]['RA'],extended_data[i][j]['DEC']))
                plt.xlabel('RA')
                plt.ylabel('DEC')
    return plt.show()