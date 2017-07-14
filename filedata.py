
# coding: utf-8

# In[2]:

import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy import io

### --- opens a file from IDL in python --- ###

def openFile(filename):                                 
    data = scipy.io.readsav(filename,python_dict=True)  #data is a dictionary with one key, 'source_array'
    return data                                         

### --- Creates a 2D density histogram of ra and dec for a single extended obj. --- ###

def histplot(data,elem):
    ra = data[elem]['extend']['ra']            # if you want to plot something directly from the
    dec = data[elem]['extend']['dec']          # variable data, use "data = data['source_array']"
    plt.hist2d(ra, dec, bins=40)
    plt.colorbar()
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.title('2-D Histogram of a Single Extended Source')
    plt.show()
    
### --- Separates data into point sources and extended objects ---------------- ###  
### --- Shows which data elements are extended, and lists their intensities --- ###

def extObj(data):
    
    data_len = range(len(data['source_array']))        #This function will not list the intensities
    point = []                                         #of an extended object's children, only the
    ext_obj = []                                       #one intensity value of the parent.

    print "Elements in this file which are extended objects:\n"
    
    for o in data_len:
        if data['source_array'][o][-2] is None:
            point.append(data['source_array'][o])
        else:
            ext_obj.append(data['source_array'][o])
            print 'element', o,#'has intensity', data['source_array'][o]['flux'][0][4]
            print '\n'

    return ext_obj 

### --- Lists all of the intensity values of each data point --- ###

def allIntensities(data):

    data_len = range(len(data['source_array']))        #This function will not list the intensities
                                                       #of an extended object's children, only the 
    for o in data_len:                                 #one intensity value of the parent.
        print 'element ' + str(o) + ' has intensity ' + str(data['source_array'][o]['flux']['i'])  

