# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:33:08 2017

@author: aaronros

Data analysis module v2: images

Current iteration of this file is only used for asc files obtained from an 
Attocube scanning probe microscopy
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import signal
import pandas as pd
import copy
import re
from constants import *

class get_data(object):
    '''Create an object of the microscopy image and data: imag_obj = getData('SC_000-Mag-fwd.asc')'''
    def __init__(self,filename):

        # check to make sure this is a .asc file
        while(True):
            if '.asc' in filename:
                while (True):
                    try:
                        break
                    except:
                        filename = input(filename + ' not found. Enter valid file: ')
                break
            else:
                filename = input('Enter .asc file: ')

        self.filename = filename

        # get the file specs such as the lengths and number of pixels        
        f = open(self.filename)
        
        for row in f:
            if '# x-pixels:' in row:
                xpixel = [int(s) for s in row.split() if s.isdigit()]
                xpixel = xpixel[0] # should only have one number in the row

            elif '# y-pixels:' in row:
                ypixel = [int(s) for s in row.split() if s.isdigit()]
                ypixel = ypixel[0] # should only have one number in the row                    

            elif '# x-length:' in row:
                xlength = [float(s) for s in re.findall("-?\d+.?\d*(?:[Ee]-\d+)?", row)]
                xlength = xlength[0]*M_TO_UM  

            elif '# y-length:' in row:
                ylength = [float(s) for s in re.findall("-?\d+.?\d*(?:[Ee]-\d+)?", row)]
                ylength = ylength[0]*M_TO_UM  

        self._specs = {'xpixel': xpixel, 'ypixel': ypixel, 'xlength':
                      xlength,'ylength': ylength, 'xlabel': 'x [$\mu$m]',
                      'ylabel': 'y [$\mu$m]', 'title': 'Scan of: '+ self.filename}
        f.close()    
    
        # initialize the data matrix
        with open(self.filename) as f:
            self.data = np.loadtxt(f)
            
    def removecol(self, colNumber):
        self._specs['xpixel'] -= 1
        microns_to_pixels = self._specs['xlength']/float(self._specs['xpixel'])
        self._specs['xlength'] -= microns_to_pixels
        self.data = sp.delete(self.data, colNumber, 1) # 1 is the dimension

    def removerow(self, rowNumber):
        self._specs['ypixel'] -= 1
        microns_to_pixels = self._specs['ylength']/float(self._specs['ypixel'])
        self._specs['ylength'] -= microns_to_pixels
        self.data = sp.delete(self.data, colNumber, 0) # 0 is the dimension

    def plotdata(self, cmap = 'RdYlGn', CS = [None,None], cbLabel = None, figNumber = 1):
        #establish x and y axes
        fig = plt.figure(figNumber, facecolor = 'white')
        x = np.linspace(float(-self._specs['xlength']) / 2, float(self._specs['xlength']) / 2,
                     self._specs['xpixel'])
        y = np.linspace(float(-self._specs['ylength']) / 2, float(self._specs['ylength']) / 2,
                     self._specs['ypixel'])                     

        # plot the figure
        plt.imshow(self.data, cmap, extent = [x[0], x[len(x) - 1], 
                                          y[0], y[len(y) - 1],], interpolation = 'none')
        cb = plt.colorbar(orientation = 'vertical')
        cb.set_label(cbLabel, fontsize = plot_font)
        plt.clim(CS[0],CS[1])        
        plt.xlabel(self._specs['xlabel'], fontsize = plot_font)
        plt.ylabel(self._specs['ylabel'], fontsize = plot_font)
        plt.title(self._specs['title'], fontsize = plot_font)
        plt.show()        
        
    def plotFFT2(self, cmap = 'BrBG', CS = [None,None], cbLabel = None, figNumber = 1):
        #establish x and y axes
        fig = plt.figure(figNumber, facecolor = 'white')
        x = np.linspace(float(-self._specs['xlength']) / 2, float(self._specs['xlength']) / 2,
                     self._specs['xpixel'])
        y = np.linspace(float(-self._specs['ylength']) / 2, float(self._specs['ylength']) / 2,
                     self._specs['ypixel'])                     

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
        
    def plotFFT2Log(self, cmap = 'BrBG', CS = [None,None], cbLabel = None, figNumber = 1):
        #establish x and y axes
        fig = plt.figure(figNumber, facecolor = 'white')
        x = np.linspace(float(-self._specs['xlength']) / 2, float(self._specs['xlength']) / 2,
                     self._specs['xpixel'])
        y = np.linspace(float(-self._specs['ylength']) / 2, float(self._specs['ylength']) / 2,
                     self._specs['ypixel'])                     

        # plot the figure
        plt.imshow(abs(np.log(np.fft.fftshift(np.fft.fft2(self.data)))), cmap, extent = [x[0], x[len(x) - 1], 
                                          y[0], y[len(y) - 1],], interpolation = 'none')
        cb = plt.colorbar(orientation = 'vertical')
        cb.set_label(cbLabel)
        plt.clim(CS[0], CS[1])
        plt.xlabel('kx', fontsize = plot_font)
        plt.ylabel('ky', fontsize = plot_font)
        plt.title('Log of frequency component', fontsize = plot_font)
        plt.show()                
              
    def globalsubtract(self):
        self.data = self.data - np.mean(self.data)
        

    def avelinesubtract(self):
        (row, col) = self.data.shape
        if col > row:
            for i in range(col):
                self.data[:,i] = self.data[:,i] - np.mean(self.data[:,i])
        else:
            for j in range(row):
                self.data[j,:] = self.data[j,:] - np.mean(self.data[j,:])
        return

    def vertlineCut(self,xpixel):
        (row, col) = self.data.shape
        y = np.linspace(float(-self._specs['ylength']) / 2,
                float(self._specs['ylength']) / 2, self._specs['ypixel'])
        dy = row - 1
        dataLine = np.zeros([dy,2])
        for i in range(dy):
            dataLine[i,0] = y[i]
            dataLine[i,1] = self.data[i,xpixel]
        return dataLine

    def horzlinecut(self,ypixel):
        (row,col) = self.data.shape
        x = np.linspace(float(-self._specs['xlength']) / 2,
                     float(self._specs['xlength']) / 2, self._specs['xpixel'])
        dx = col - 1
        dataLine = np.zeros([dx,2])
        for i in range(dx):
            dataLine[i,0] = x[i]
            dataLine[i,1] = self.data[ypixel,i]
        return dataLine
        
    def get_hist(self, binSize = 100):
        scan_objD = self.data.ravel() # need 1D array for hist
        fig = plt.figure(facecolor = 'white')
        histValue = plt.hist(scan_objD, bins = 100)
        
        histReturn = [[], []]        
        
        histReturn[0] = histValue[1][0:len(histValue[1]) -1]
        histReturn[1] = histValue[0]
        
        plt.xlabel('Insert Value', fontsize = plot_font, labelpad=10)
        plt.ylabel('Number of Counts', fontsize = plot_font, labelpad=10)
        strTitle = 'Histogram of ' + self.filename
        plt.title(strTitle, fontsize = plot_font)    
        
        return histReturn
        
    def median_filter(self):
        xpixel = self._specs['xpixel']
        ypixel = self._specs['ypixel']                          
        for i in range(1,int(xpixel)-1):
            for j in range(1,int(ypixel)-1):
                ptr_cen = self.data[i,j]
                i_near = [-1,1]
                j_near = [-1, 1]
                pts_vec = [ptr_cen]
                for pts_horz in i_near:
                    for pts_vert in j_near:
                        pts_vec.append(self.data[i+pts_horz, j+pts_vert])
                self.data[i,j] = np.median(pts_vec)
        return 
     
class get_mag (getData):

    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self._specs['title'] = 'Magnetometry scan: ' + fileName

class get_dPhi_dI (getData):

    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = self.data/(VtoPhi * gainLockAux2 * I_sample)
        self._specs['title'] = 'd$\Phi$/dI scan: ' + fileName

class get_dPhi_dV (getData):

    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = self.data/(VtoPhi * gainLockAux2 * Vapplied)
        self._specs['title'] = 'd$\Phi$/dV scan: ' + fileName
      
        
class get_susc (getData):

    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = self.data/(VtoPhi * gainDC * I_FC)
        self._specs['title'] = 'Susc scan: ' + fileName


class get_suscy (getData):

    def __init__(self,fileName):
        getData.__init__(self,fileName)
        self.globalSubtract()
        self.data = self.data/(VtoPhi * gainDC * I_sample)
        self._specs['title'] = 'Suscy scan: ' + fileName
        return

class get_cap (get_data):

    def __init__(self,filename):
        getData.__init__(self,filename)
        self.data = self.data * V_TO_mV
        self._specs['title'] = 'Capacitance scan: ' + fileName
    

'''
General functions for data analysis
'''

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

'''
Function: align_scans
Usage: mat_diff, x_best, y_best = align_scans(mag_obj1, mag_obj2)
---
Given two image objects that are nominally identical, this functions
finds how much to shift scan_obj2 relative to scan_obj1 to get the 
correct alignment. This function returns two matrices for each of the shifted
'''
def align_scans(mag_obj1, mag_obj2, inplace = False, plot_corr = False,
                pnt_text = True):  
    
    if pnt_text:
        print('align_scans function called. Cross correlation in progress')
    
    x_shape = mag_obj1.specs['xpixel']
    y_shape = mag_obj1.specs['ypixel']

    # self correlation to find the center point of the 2D correlation image
    corr = signal.correlate2d(mag_obj1.data, mag_obj1.data)
    y_self, x_self = np.unravel_index(np.argmax(corr), corr.shape)
    
    corr = signal.correlate2d(mag_obj1.data, mag_obj2.data)
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    y_shift = y_self - y
    x_shift = x_self - x
    
    if pnt_text:
        print('y_shift:' + str(y_shift))
        print('x_shift:' + str(x_shift))    
    
    # copy matrices
    M1 = copy.deepcopy(mag_obj1.data)
    M2 = copy.deepcopy(mag_obj2.data)

    # remove rows in y    
    if x_shift > 0: 
        # kill off first yshift rows of shifted image
        M2 = M2[:, x_shift:]
        M1 = M1[:, :y_shape - x_shift]            
    elif x_shift < 0:
        # kill last yshift rows of shifted image
        M2 = M2[:, :y_shape - x_shift]
        M1 = M1[:, x_shift:]            

    # remove rows in x
    if y_shift > 0:
        # kill off first xshift rows of shifted image
        M2 = M2[y_shift:, :]
        M1 = M1[:x_shape - y_shift, :]
        
    elif y_shift < 0:
        # kill off first xshift rows of shifted image
        M2 = M2[:x_shape - y_shift, :]
        M1 = M1[y_shift:, :]
                
    if plot_corr:
        fig = plt.figure(facecolor = 'white')
        plt.imshow(corr, cmap = 'RdYlGn')
        plt.xlabel('', fontsize = plot_font)
        plt.ylabel('', fontsize = plot_font)        
        plt.title('Correlation between images', fontsize = plot_font)        

    if inplace:
        mag_obj1.data = M1
        mag_obj2.data = M2   
    else:
        return M1, M2