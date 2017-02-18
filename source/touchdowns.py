# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:01:40 2017

@author: aaronros

Data analysis module v2: touchdowns
"""

import numpy as np
import matplotlib.pyplot as plt # change to include the decorator
import pandas as pd
from constants import *

SKIP_ROWS = 7 # attocute software has 7 excess rows that we remove


'''
define class: getTD
 
Because the touchdowns all have the same data loading type, we define a
general class that loads such files.

Updated to include exceptions
'''
class get_TD (object):
    '''Create a touchdown object for plotting and anaylsis: td_obj = getTD('S1_000-Capacitance.csv')'''
    def __init__(self,filename):

        while(True):
            if filename[len(filename) - 1] == 'v':
                while (True):
                    try:
                        df= pd.read_csv(filename, skiprows=SKIP_ROWS, sep = ';', 
                                        header= None, names = ['pos', 'data'])
                        break
                    except:
                        filename = input('Enter valid file: ')
                break
            else:
                print('Error: Not a .csv file')
                filename = input('Enter .csv file: ')
                
        pos = np.array(df.pos) * M_TO_UM
        data = np.array(df.data)
        
        self.pos = pos
        self.data = data
        self.filename = filename

        # subtract the offset from the data. We take the mean of the first 
        # 10 points as the offself. If there are less than 10 points in the 
        # touchdown file we just subtract the first point from the mean
        try:
            self.data = self.data - np.mean(self.data[:10])
        except:
            self.data = self.data - np.mean(self.data[0])

        self._specs = {'xlabel': 'Position [$\mu$m]', 'ylabel': 'Data [V]',
                       'title': 'Touchdown file: '+ self.filename}

    def plotTD(self, figNumber = 1):
        fig = plt.figure(figNumber, facecolor = 'white')
        plt.plot(self.pos, self.data, 'bo-')
        plt.xlabel(self._specs['xlabel'], fontsize = plot_font)
        plt.ylabel(self._specs['ylabel'], fontsize = plot_font)
        plt.title(self._specs['title'], fontsize = plot_font)
        plt.show()
        
    def __str__(self):
        return 'Touchdown file: ' + self.filename ,str(self.pos), str(self.data)
        
    
class get_capTD (getTD):

    def __init__(self,filename):
        getTD.__init__(self,filename)
        self.data = self.data * V_TO_mV 
        self._specs['ylabel'] = 'Capacitance [mV]'
        self._specs['title'] = 'Capacitance TD file: '+ self.filename
 
class get_magTD (getTD):

    def __init__(self,filename):
        getTD.__init__(self,filename)
        self.data = self.data / (VtoPhi * gainDC) * PHI0_TO_mPHI0
        self._specs['ylabel'] = 'Magnetometry [m$\Phi_0$]'
        self._specs['title'] = 'Magnetometry TD file: '+ self.filename

  
class get_suscTD (getTD):
    def __init__(self,filename):
        getTD.__init__(self,filename)
        self.data = self.data / (VtoPhi * gainLockAux2 * I_FC) 
        self._specs['ylabel'] = 'Susceptibility [$\Phi_0$/A]'
        self._specs['title'] = 'Susceptibility TD file: '+ self.filename


class get_currTD (getTD):
    def __init__(self,filename):
        getTD.__init__(self,filename)
        self.data = self.data / (VtoPhi * gainLockAux2 * I_sample) 
        self._specs['ylabel'] = 'Current [$\Phi_0$/A]'
        self._specs['title'] = 'Current TD file: '+ self.filename
    
  