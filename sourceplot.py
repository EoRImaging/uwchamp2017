import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from scipy import io
import glob

def readandplot(location):
    '''Description'''
        
        ###==================================================================###
        ###------------------------------------------------------------------###
        ###--------------Collecting All Data Files from Folder---------------###
        ###------------------------------------------------------------------###
        ###==================================================================###
    
    data = []
    all_files = glob.glob(location)
    for i in range(len(all_files)):
        data.append(scipy.io.readsav(all_files[i],python_dict=True))
        
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
    
    for i in range(len(all_files)):
        
        ###==================================================================###
        ###--------------------Initializing nested lists---------------------###
        ###==================================================================###
        
        extended_data_nest = []
        point_data_nest = []
        extended_X_nest = []
        extended_Y_nest = []
        extended_FREQ_nest = []
        extended_I_nest = []
        
        ###==================================================================###
        ###-----------------------Creating nested lists----------------------###
        ###==================================================================###
        
        for j in range(len(data[i]['source_array'])):
            if data[i]['source_array'][j][-2] is None:
                point_data_nest.append(data[i]['source_array'][j])
            else:
                extended_data_nest.append(data[i]['source_array'][j])
        
        ###==================================================================###
        ###------------------Creating fully separated lists------------------###
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
        
        ###==================================================================###
        ###------------------------------------------------------------------###
        ###------------------Plotting All Extended Sources-------------------###
        ###------------------------------------------------------------------###
        ###==================================================================###
        
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

