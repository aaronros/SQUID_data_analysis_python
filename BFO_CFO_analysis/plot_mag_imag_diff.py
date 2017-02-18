'''
Plot raw magnetometry images

Created: 2017-01-31

Desc: Loads the raw magnetometry data and plots it
'''

import sys
sys.path.append('../source/')
import data_analysis_module as da

import matplotlib.pyplot as plt
import numpy as np

# global constants
PHI0_TO_MPHI0 = 1000
COLOR_SCALE = [-16,16]
da.set_VtoPhi(0.332)
da.set_GainDC(10)

plt.close('all')


def plot_mag_mean(M_vec, V_vec):

    means = []
    
    mag_obj_org = da.getMag(M_vec[0])
    mag_obj_org.data = mag_obj_org.data * PHI0_TO_MPHI0
        
    for i in range(1, len(M_vec)):    
        print('Image number: ' + str(i))
        mag_obj_tmp = da.getMag(M_vec[i])
        mag_obj_tmp.data = mag_obj_tmp.data * PHI0_TO_MPHI0 
        
        M1, M2 = da.align_scans(mag_obj_org, mag_obj_tmp)
        
        if i == 1:
            means.append(np.mean(M1))
        
        means.append(np.mean(M2))

        mag_obj_tmp.toString['title'] = 'Magnetometry Scan after $V_{bg} = $' + str(V_vec[i]) + ' V'
        mag_obj_tmp.plotData(cbLabel= 'm$\Phi_0$', figNumber= i, CS= COLOR_SCALE)

        
    fig1 = plt.figure(facecolor = 'white')
    plt.plot(V_vec, means, 'bo')
    plt.title('Mean of Images', fontsize = 25)
    plt.xlabel('Back gate voltage [V]', fontsize = 25)     
    plt.ylabel('$\sigma_{Mag}$', fontsize = 25)
    plt.show()
    
    return means
    
def gaussian_fits(M_vec, V_vec):

    diff_images = []
    V_diff = []
        
    for i in range(1, len(M_vec)):    
        print('Image number: ' + str(i))
        
        mag_obj_org = da.getMag(M_vec[i - 1])
        mag_obj_org.data = mag_obj_org.data * PHI0_TO_MPHI0

        mag_obj_tmp = da.getMag(M_vec[i])
        mag_obj_tmp.data = mag_obj_tmp.data * PHI0_TO_MPHI0 

        M1, M2 = da.align_scans(mag_obj_org, mag_obj_tmp)
        
        diff = M1 - M2
         
        diff_images.append(diff)
        V_diff.append(V_vec[i] - V_vec[i - 1])

    std_vec = []

    for i in range(len(diff_images)):
        mag_obj_tmp.data = diff_images[i]
        mag_obj_tmp.aveLineSubtract()
        histReturn = mag_obj_tmp.getHist()    
        (popt, pcov) = da.fitGauss(histReturn[0], histReturn[1], [.1, .1, .1])  
        std_vec.append(popt[2])
        mag_obj_tmp.toString['title'] = 'Difference for V =  ' + str(V_diff[i]) + ' V'
        mag_obj_tmp.plotData(figNumber = i)
        
    fig1 = plt.figure(facecolor = 'white')  
    plt.plot(V_diff, std_vec, 'bo')
    plt.xlabel('$\Delta V_{bg}$', fontsize = 16)
    plt.ylabel('$\sigma_{Mag}$', fontsize = 16)
    plt.title('Standard deviation of the magnetometry images difference between $V_{bg}$',fontsize = 14)
        
    return std_vec
 
 
 
def main():
    M_org = 'SC_069-Mag-fwd.asc'
    M_p2 = 'SC_070-Mag-fwd.asc'
    M_n4 = 'SC_071-Mag-fwd.asc'
    M_p6 = 'SC_072-Mag-fwd.asc'
    M_n10 = 'SC_073-Mag-fwd.asc'
    M_p20 = 'SC_074-Mag-fwd.asc'
    M_n30 = 'SC_075-Mag-fwd.asc'
    
    M_vec = [M_org, M_p2, M_n4, M_p6, M_n10, M_p20, M_n30] # vector of mag files
    V_vec = [0, 2, -4, 6, -10, 20, -30] # corresponding vector of gate voltages

    
    plot_mag_mean(M_vec, V_vec)
    #gaussian_fits(M_vec, V_vec)
   
if __name__ == '__main__':
    main()
    




