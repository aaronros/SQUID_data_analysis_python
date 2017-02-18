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
            
    def removeCol(self, colNumber):
        self._specs['xpixel'] -= 1
        microns_to_pixels = self._specs['xlength']/float(self._specs['xpixel'])
        self._specs['xlength'] -= microns_to_pixels
        self.data = sp.delete(self.data, colNumber, 1) # 1 is the dimension

    def removeRow(self, rowNumber):
        self._specs['ypixel'] -= 1
        microns_to_pixels = self._specs['ylength']/float(self._specs['ypixel'])
        self._specs['ylength'] -= microns_to_pixels
        self.data = sp.delete(self.data, colNumber, 0) # 0 is the dimension

    def plotData(self, cmap = 'RdYlGn', CS = [None,None], cbLabel = None, figNumber = 1):
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
        return
        
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
        return  
        
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
        return          
              
    def globalSubtract(self):
        self.data = self.data - np.mean(self.data)
        

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
        y = np.linspace(float(-self._specs['ylength']) / 2,
                float(self._specs['ylength']) / 2, self._specs['ypixel'])
        dy = row - 1
        dataLine = np.zeros([dy,2])
        for i in range(dy):
            dataLine[i,0] = y[i]
            dataLine[i,1] = self.data[i,xpixel]
        return dataLine

    def horzLineCut(self,ypixel):
        (row,col) = self.data.shape
        x = np.linspace(float(-self._specs['xlength']) / 2,
                     float(self._specs['xlength']) / 2, self._specs['xpixel'])
        dx = col - 1
        dataLine = np.zeros([dx,2])
        for i in range(dx):
            dataLine[i,0] = x[i]
            dataLine[i,1] = self.data[ypixel,i]
        return dataLine
        
    def getHist(self, binSize = 100):
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
     
          
