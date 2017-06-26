import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import io
import glob

def collector(path):
    '''Collects all IDL save data from a given path and stores each file as an element in a list.'''
        
    ###==================================================================###
    ###------------------------------------------------------------------###
    ###--------------Collecting All Data Files from Folder---------------###
    ###------------------------------------------------------------------###
    ###==================================================================###
    
    filenames = glob.glob(path)
    
    data = [[scipy.io.readsav(filenames[i],python_dict=True) for i in range(len(filenames))],filenames]
    return data

def separator(data):
    '''Compiles data into separate lists of extended and point sources'''
    ###==================================================================###
    ###------------------------------------------------------------------###
    ###--------------Separating Point and Extended Sources---------------###
    ###------------------------------------------------------------------###
    ###==================================================================###

    ###==================================================================###
    ###----------------Initializing comprehensive lists------------------###
    ###==================================================================###
    
    point_data = []
    extended_data = []
    extended_X = []
    extended_Y = []
    extended_FREQ = []
    extended_I = []
    
    for i in range(len(data[0])):
        
        ###==================================================================###
        ###---------------Initializing nested separated lists----------------###
        ###==================================================================###
        
        extended_data_nest = []
        point_data_nest = []
        extended_X_nest = []
        extended_Y_nest = []
        extended_FREQ_nest = []
        extended_I_nest = []
        
        ###==================================================================###
        ###------------------Creating nested separated lists-----------------###
        ###==================================================================###
        
        for j in range(len(data[0][i]['source_array'])):
            if data[0][i]['source_array'][j][-2] is None:
                point_data_nest.append(data[0][i]['source_array'][j])
            else:
                extended_data_nest.append(data[0][i]['source_array'][j])
        
        ###==================================================================###
        ###--------------Creating comprehensive separated lists--------------###
        ###==================================================================###
        
        point_data.append(point_data_nest)
        extended_data.append(extended_data_nest)
        
        ###==================================================================###
        ###------Creating nested lists for parameters of extended sources----###
        ###==================================================================###
        
        for j in range(len(extended_data[i])):
            extended_X_nest.append(extended_data[i][j]['X'])
            extended_Y_nest.append(extended_data[i][j]['Y'])
            extended_FREQ_nest.append(extended_data[i][j]['FREQ'])
            extended_I_nest.append(extended_data[i][j]['FLUX']['I'][0])
        
        ###==================================================================###
        ###------------Appending nested lists to comprehensive list----------###
        ###==================================================================###
        
        extended_X.append(extended_X_nest)
        extended_Y.append(extended_Y_nest)
        extended_FREQ.append(extended_FREQ_nest)
        extended_I.append(extended_I_nest)

    return extended_data
    #return point_data
    #return extended_X
    #return extended_Y
    #return extended_FREQ
    #return extended_I

def plotall(data):
    ###==================================================================###
    ###------------------------------------------------------------------###
    ###------------------Plotting All Extended Sources-------------------###
    ###------------------------------------------------------------------###
    ###==================================================================###
    
    separator(data)
    
    for i  in range(len(data)):
        plt.figure(figsize=(18,15))
        plt.suptitle('{}'.format(all_files[i]),fontsize=20)
        
        plt.subplot(3,3,1)
        plt.scatter(extended_X[i],extended_Y[i],s=10,c=extended_FREQ[i])
        plt.title('Frequency (Only Extended Sources)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        
        plt.subplot(3,3,2)
        plt.scatter(extended_X[i],extended_Y[i],s=10,c=np.fft.ifft(extended_I[i]),vmax=.3)
        plt.title('Inverse FT Intensity (Only Extended Sources)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        
        plt.subplot(3,3,3)
        plt.hist2d(extended_X[i],extended_Y[i],bins=100)
        plt.title('Position (Only Extended Sources)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        
        plt.show()

def plotext(data,minI):
    
    extended_data = separator(data)
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