'''
data_anaylsis_module.py

Created to interpret raw data from a scanning SQUID microscope
and allows the client to perform data analysis with built-in
routines. 

The breakdown of this file is as follows:
    - initialize global variables and write helperfunctions
    to get and set these variables. These are constants
    such as the flux to voltage conversion and dc gain

    - create touchdown object that interprets the touchdown data
    allowing the user to process and plot the data

    - create image object that interprets 2D images (matrices)
    allowing the user to processa nd plot the data

IMPROVEMENTS
    - use pandas to interpret touchdown data
    - create multiple modules instead of one

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd


'''
constants
'''
M_TO_UM = 10**6 # microns
V_TO_mV = 10**3 # mV

'''
Get and set global variables 
'''

VtoPhi = 0.332 # SQUID periodicity
gainDC = 10 # gain of magnetometry
gainLockAux1 = 10000 #gain of aux1
gainLockAux2 = 100 #gain of aux2
gainLockAux3 = 100 #gain of aux2
I_FC = 0.5e-3
Vapplied = 1
I_sample = 0.5e-3
plot_font = 16

def set_VtoPhi(value):
    global VtoPhi
    VtoPhi = value
    return

def set_GainDC(value):
    global gainDC
    gainDC = value
    return

def set_GainLockAux1(value):
    global gainLockAux1
    gainLockAux1 = value
    return

def set_GainLockAux2(value):
    global gainLockAux2
    gainLockAux2 = value
    return

def set_GainLockAux3(value):
    global gainLockAux3
    gainLockAux3 = value
    return

def set_I_sample (value):
    global I_sample
    I_sample = value
    return

def set_I_FC(value):
    global I_FC
    I_FC = value

def set_V_sample(value):
    global V_applied
    V_applied = value

def set_plot_font(value):
    global plot_font
    plot_font = value

def get_VtoPhi():
    global VtoPhi
    return VtoPhi

def get_GainDC():
    global gainDC
    return gainDC

def get_GainLockAux1():
    global gainLockAux1
    return gainLockAux1

def get_GainLockAux2():
    global gainLockAux2
    return gainLockAux2

def get_GainLockAux3():
    global gainLockAux3
    return gainLockAux3

def get_I_sample():
    global I_sample
    return I_sample
    
def get_plot_font():
    global plot_font
    return plot_font
 
        

'''
 define class: getTD
 Created 10/24/2014

 Because the touchdowns all have the same data loading type, we define a
 general class that loads such files.

 Updated to include exceptions
'''
class getTD:

    # method: constructor
    # Created 10/24/2014
    #
    # Requires fileName. This method will load the touchdown file.
    def __init__(self,filename):

        # initialize arrays to use for this program. Note that position and
        # capacitance have initial starting points of 0. These data points will
        # be removed at the end of the code.
    
        while(True):
            if filename[len(filename) - 1] == 'v':
                while (True):
                    try:
                        df= pd.read_csv(filename, skiprows=7, sep = ';', 
                                        header= None, names = ['x', 'y'])
                        break
                    except IOError as e:
                        print "I/O error({0}): {1}".format(e.errno, e.strerror)
                        fileName = input('Enter valid file: ')
                break
            else:
                print 'Error: Not a .csv file'
                filename = input('Enter .csv file: ')
                


    
        x = np.array(df.x) * M_TO_UM
        y = np.array(df.y)
        
        self.x = x
        self.y = y
        self.toString = {'xlabel': 'position [$\mu$m]', 'ylabel':
                         'y', 'title': 'Touchdown file of '
                         + filename}


    def plotTD(self, figNumber = 1):
        fig = plt.figure(figNumber, facecolor = 'white')
        plt.plot(self.x, self.y, 'bo-')
        plt.xlabel(self.toString['xlabel'], fontsize = plot_font)
        plt.ylabel(self.toString['ylabel'], fontsize = plot_font)
        plt.title(self.toString['title'], fontsize = plot_font)
        plt.show()

# define class: getCapTD
# Created 10/25/2014
#
# Extends getTD class, specifically for capacitance
class getCapTD (getTD):

    # method: constructor
    # Created 10/25/2014
    #
    # Requires fileName. This method will load the capacitance file.
    def __init__(self,fileName):
        getTD.__init__(self,fileName)
        fileNumber = fileName[3:6]
        self.y = self.y * V_TO_mV
        self.toString = {'xlabel': 'position [$\mu$m]', 'ylabel':
                         'Capacitance [mV]', 'title': 'Capacitance touchdown '
                         + fileNumber}        

# define class: getMagTD
# Created 10/25/2014
#
# Extends getTD class, specifically for magnetometry
class getMagTD (getTD):

    # method: constructor
    # Created 10/25/2014
    #
    # Requires fileName. This method will load the magnetometry file.

    # Note! No background subtraction on this file
    def __init__(self,fileName):
        getTD.__init__(self,fileName)
        fileNumber = fileName[3:6]
        self.y = self.y / (VtoPhi * gainDC)
        self.toString = {'xlabel': 'position [$\mu$m]', 'ylabel':
                         'Magnetometry [$\Phi$/$\Phi_0$]', 'title': 'Magnetomtry touchdown: file number '
                         + fileNumber}                             

# define class: getSuscTD
# Created 10/25/2014
#
# Extends getTD class, specifically for capacitance
class getSuscTD (getTD):

    # method: constructor
    # Created 10/25/2014
    #
    # Requires fileName. This method will load the magnetometry file.
    def __init__(self,fileName):
        getTD.__init__(self,fileName)
        fileNumber = fileName[3:6]
        self.y = self.y / (VtoPhi * gainLockAux2 * I_FC)
        self.toString = {'xlabel': 'position [$\mu$m]', 'ylabel':
                         'Susceptibility [$\Phi_0$/A]', 'title': 'Susceptibility touchdown: file number '
                         + fileNumber}        

class getCurrTD (getTD):
    # method: constructor
    # Created 10/25/2014
    #
    # Requires fileName. This method will load the magnetometry file.
    def __init__(self,fileName):
        getTD.__init__(self,fileName)
        fileNumber = fileName[3:6]
        self.y = self.y / (VtoPhi * gainLockAux2 * I_sample)
        self.toString = {'xlabel': 'position [$\mu$m]', 'ylabel':
                         'ac Current [$\Phi_0$/A]', 'title': 'ac Current touchdown: file number '
                         + fileNumber}         

# define class: getData
# Created 10/29/2014
#
# Initial class to create an object of a matrix of data
#
# Updated to include exceptions
class getData:

    def __init__(self,fileName):

        # initialize length and pixels


        while(True):
            if fileName[len(fileName) - 1] == 'c':
                while (True):
                    try:
                        dataFile = open(fileName)
                        break
                    except IOError as e:
                        print "I/O error({0}): {1}".format(e.errno, e.strerror)
                        fileName = input('Enter valid file: ')
                break
            else:
                print 'Error: Not a .asc file'
                fileName = input('Enter .asc file: ')

        for row in dataFile:
            numChars = 11
            if row[0:numChars] == '# x-pixels:':
                numChars += 2
                temp = row[numChars]
                while (1):
                    numChars += 1
                    if numChars == len(row):
                        break
                    else:
                        temp = temp + row[numChars] #concat strings
                xpixel = float(temp)

            elif row[0:numChars] == '# y-pixels:':
                numChars += 2
                temp = row[numChars]
                while (1):
                    numChars += 1
                    if numChars == len(row):
                        break
                    else:
                        temp = temp + row[numChars] #concat strings
                ypixel = float(temp)

            elif row[0:numChars] == '# x-length:':
                numChars += 2
                temp = row[numChars]
                while (1):
                    numChars += 1
                    if numChars == len(row):
                        break
                    else:
                        temp = temp + row[numChars] #concat strings
                xlength = float(temp)

            elif row[0:numChars] == '# y-length:':
                numChars += 2
                temp = row[numChars]
                while (1):
                    numChars += 1
                    if numChars == len(row):
                        break
                    else:
                        temp = temp + row[numChars] #concat strings
                ylength = float(temp)


        self.specs = {'xpixel': xpixel, 'ypixel': ypixel, 'xlength':
                      xlength * 10**6,'ylength': ylength * 10**6}
        self.toString = {'xlabel': r'x [$\mu$m]', 'ylabel':
                         'y [$\mu$m]', 'title': 'Data File of '
                         + fileName}
        

        # initialize the data matrix
        self.data = np.loadtxt(fileName)

    
        
    def removeCol(self, colNumber):
        return sp.delete(self, colNumber, 1) # 1 is the dimension

    def removeRow(self, rowNumber):
        return sp.delete(self, rowNumber, 0) # 0 is the dimension

    def plotData(self, cmap = 'RdYlGn', CS = [None,None], cbLabel = None, figNumber = 1):
        #establish x and y axes
        fig = plt.figure(figNumber, facecolor = 'white')
        x = np.linspace(float(-self.specs['xlength']) / 2, float(self.specs['xlength']) / 2,
                     self.specs['xpixel'])
        y = np.linspace(float(-self.specs['ylength']) / 2, float(self.specs['ylength']) / 2,
                     self.specs['ypixel'])                     

        # plot the figure
        plt.imshow(self.data, cmap, extent = [x[0], x[len(x) - 1], 
                                          y[0], y[len(y) - 1],], interpolation = 'none')
        cb = plt.colorbar(orientation = 'vertical')
        cb.set_label(cbLabel, fontsize = plot_font)
        plt.clim(CS[0],CS[1])        
        plt.xlabel(self.toString['xlabel'], fontsize = plot_font)
        plt.ylabel(self.toString['ylabel'], fontsize = plot_font)
        plt.title(self.toString['title'], fontsize = plot_font)
        plt.show()        
        return
        
    def plotFFT2(self, cmap = 'BrBG', CS = [None,None], cbLabel = None, figNumber = 1):
        #establish x and y axes
        fig = plt.figure(figNumber, facecolor = 'white')
        x = np.linspace(float(-self.specs['xlength']) / 2, float(self.specs['xlength']) / 2,
                     self.specs['xpixel'])
        y = np.linspace(float(-self.specs['ylength']) / 2, float(self.specs['ylength']) / 2,
                     self.specs['ypixel'])                     

        # plot the figure
        plt.imshow(abs((np.fft.fftshift(np.fft.fft2(self.data)))), cmap, extent = [x[0], x[len(x) - 1], 
                                          y[0], y[len(y) - 1],], interpolation = 'none')
        cb = plt.colorbar(orientation = 'vertical')
        cb.set_label(cbLabel)
        plt.clim(CS[0], CS[1])
        plt.xlabel('kx', fontsize = plot_font)
        plt.ylabel('ky', fontsize = plot_font)
        plt.title('Frequency Component', fontsize = plot_font)
        plt.show()        
        return  
        
    def plotFFT2Log(self, cmap = 'BrBG', CS = [None,None], cbLabel = None, figNumber = 1):
        #establish x and y axes
        fig = plt.figure(figNumber, facecolor = 'white')
        x = np.linspace(float(-self.specs['xlength']) / 2, float(self.specs['xlength']) / 2,
                     self.specs['xpixel'])
        y = np.linspace(float(-self.specs['ylength']) / 2, float(self.specs['ylength']) / 2,
                     self.specs['ypixel'])                     

        # plot the figure
        plt.imshow(abs(np.log(np.fft.fftshift(np.fft.fft2(self.data)))), cmap, extent = [x[0], x[len(x) - 1], 
                                          y[0], y[len(y) - 1],], interpolation = 'none')
        cb = plt.colorbar(orientation = 'vertical')
        cb.set_label(cbLabel)
        plt.clim(CS[0], CS[1])
        plt.xlabel('kx', fontsize = plot_font)
        plt.ylabel('ky', fontsize = plot_font)
        plt.title('Frequency Component', fontsize = plot_font)
        plt.show()        
        return          
                

    def globalSubtract(self):
        self.data = self.data - np.mean(self.data)
        return

    def aveLineSubtract(self):
        (row, col) = self.data.shape
        if col > row:
            for i in range(col):
                self.data[:,i] = self.data[:,i] - np.mean(self.data[:,i])
        else:
            for j in range(row):
                self.data[j,:] = self.data[j,:] - np.mean(self.data[j,:])
        return

    def vertLineCut(self,xpixel):
        (row, col) = self.data.shape
        y = np.linspace(float(-self.specs['ylength']) / 2,
                float(self.specs['ylength']) / 2, self.specs['ypixel'])
        dy = row - 1
        dataLine = np.zeros([dy,2])
        for i in range(dy):
            dataLine[i,0] = y[i]
            dataLine[i,1] = self.data[i,xpixel]
        return dataLine

    def horzLineCut(self,ypixel):
        (row,col) = self.data.shape
        x = np.linspace(float(-self.specs['xlength']) / 2,
                     float(self.specs['xlength']) / 2, self.specs['xpixel'])
        dx = col - 1
        dataLine = np.zeros([dx,2])
        for i in range(dx):
            dataLine[i,0] = x[i]
            dataLine[i,1] = self.data[ypixel,i]
        return dataLine
        
    def getHist(self, binSize = 100):
        data1D = self.data.ravel() # need 1D array for hist
        fig = plt.figure(100, facecolor = 'white')
        histValue = plt.hist(data1D, bins = 100)
        
        histReturn = [[], []]        
        
        histReturn[0] = histValue[1][0:len(histValue[1]) -1]
        histReturn[1] = histValue[0]
        
        plt.xlabel('Insert Value', fontsize = plot_font, labelpad=10)
        plt.ylabel('Number of Counts', fontsize = plot_font, labelpad=10)
        strTitle = 'Histogram of ' + self.toString['title']
        plt.title(strTitle, fontsize = plot_font)    
        
        return histReturn
        
    def median_filter(self):
        xpixel = self.specs['xpixel']
        ypixel = self.specs['ypixel']                          
        for i in range(1,int(xpixel)-1):
            for j in range(1,int(ypixel)-1):
                ptr_cen = self.data[i,j]
                i_near = [-1,1]
                j_near = [-1, 1]
                pts_vec = [ptr_cen]
                for pts_horz in i_near:
                    for ptrs_vert in j_near:
                        pts_vec.append(self.data[i+pts_horz, j+pts_vec])
                self.data[i,j] = np.median(pts_vec)
        return 
                

# define class: getMag
# Created 11/1/2014
#
# Extends getData class, specifically for magnetometry
class getMag (getData):

    # method: constructor
    # Created 11/1/2014
    #
    # Requires fileName. This method will load the magnetometry file.
    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = scale_2D_list(self.data, float(1)/(VtoPhi * gainDC))
        self.toString = {'xlabel': r'x [$\mu$m]', 'ylabel':
                         'y [$\mu$m]', 'title': 'Magnetometry File of ' + fileName}
        return

# define class: get_dPhi_dI
# Created 08/13/2015

#
# Extends getData class, specifically for dPhi/dI
class get_dPhi_dI (getData):

    # method: constructor
    # Created 08/03/2015
    #
    # Requires fileName. This method will load the magnetometry file.
    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = scale_2D_list(self.data, float(1)/(VtoPhi * gainLockAux2 * I_sample))
        self.toString = {'xlabel': r'x [$\mu$m]', 'ylabel':
                         'y [$\mu$m]', 'title': 'dPhi/dI File of ' + fileName}
        return

# define class: get_dPhi_dI
# Created 08/13/2015

#
# Extends getData class, specifically for dPhi/dI
class get_dPhi_dV (getData):

    # method: constructor
    # Created 08/03/2015
    #
    # Requires fileName. This method will load the magnetometry file.
    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = scale_2D_list(self.data, float(1)/(VtoPhi * gainLockAux2 * Vapplied))
        self.toString = {'xlabel': r'x [$\mu$m]', 'ylabel':
                         'y [$\mu$m]', 'title': 'dPhi/dV File of ' + fileName}
        return
      
        
# define class: getSusc
# Created 11/4/2014
#
# Extends getData class, specifically for susceptibility
class getSusc (getData):

    # method: constructor
    # Created 11/4/2014
    #
    # Requires fileName. This method will load the magnetometry file.
    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = scale_2D_list(self.data, float(1)/(VtoPhi * gainDC * I_FC))
        self.toString = {'xlabel': r'x [$\mu$m]', 'ylabel':
                         'y [$\mu$m]', 'title': 'Susceptibility File of ' + fileName}
        return



# define class: getSuscy
# Created 11/4/2014
#
# Extends getData class, specifically for out of phase susceptibility
class getSuscy (getData):

    # method: constructor
    # Created 11/4/2014
    #
    # Requires fileName. This method will load the magnetometry file.
    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = self.scale_2D_list(self.data, float(1)/(VtoPhi * gainDC * I_sample))
        self.toString = {'xlabel': r'x [$\mu$m]', 'ylabel':
                         'y [$\mu$m]', 'title': 'Out-of-Phase Susceptibility File of ' + fileName}
        return
    

'''
General functions for data analysis
'''

def scale_list(vec, scale):
    for i in range(len(vec)):
        vec[i] = vec[i] * scale
    return vec

def scale_2D_list(matrix, scalar):
    (row, col) = matrix.shape
    for i in range(row):
        for j in range(col):
            matrix[i,j] = matrix[i,j] * scalar
    return matrix
    
def funcGauss(x, a, x0, sigma):
    # x: discretized xaxis used as the independent value
    # a: amplitude
    # x0: peak values of gaussian function
    # sigma: sigma
    return a * np.exp(-(x-x0)**2/(2*sigma**2))
    
def fitGauss(xdata, ydata, guess = None):
    popt, pcov = opt.curve_fit(funcGauss, xdata, ydata, p0 = guess)
    popt[2] = abs(popt[2]) # want the sigma to be positive
    return (popt, pcov)    

def shiftMatrixHorz(orig_matrix,I,J,n):
    new_matrix = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            if (j+n < J):
                new_matrix[int(i),int(j)] = orig_matrix[int(i),int(j+n)] 
            else: 
                new_matrix[int(i),int(j)] = orig_matrix[int(i),int(j+n-J)]             
    return new_matrix
  
def shiftMatrixVert(orig_matrix,I,J,m):
    new_matrix = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            if (i+m < I):
                new_matrix[int(i),int(j)] = orig_matrix[int(i+m), int(j)] 
            else: 
                new_matrix[int(i),int(j)] = orig_matrix[int(i+m-I),int(j)]             
    return new_matrix  
