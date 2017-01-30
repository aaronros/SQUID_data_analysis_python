'''
Plot difference in magnetometry images

Created: 2017-01-25

Desc: Loads the raw magnetometry data and subtracts the different images
Does this for all combinations of raw data

'''

import data_analysis_module as da
import chi_squared_mod as cq
import matplotlib.pyplot as plt
import numpy as np
import itertools


# global constnats
PHI0_TO_MPHI0 = 1000
COLOR_SCALE = [-1.5,1.5]
da.set_VtoPhi(0.332)
da.set_GainDC(10)

plt.close('all')

'''
Function: order
Usage: subset[0], subset[1] = order( (1,3) )
---
For a given subset of images, order the subset such that the 
the larger one comes first
'''
def order(subset):
    if subset[0] < subset[1]:
        return (subset[1], subset[0])
    return (subset[0], subset[1])

def gaussian_fits():

    M_org = 'SC_069-Mag-fwd.asc'
    M_p2 = 'SC_070-Mag-fwd.asc'
    M_n4 = 'SC_071-Mag-fwd.asc'
    M_p6 = 'SC_072-Mag-fwd.asc'
    M_n10 = 'SC_073-Mag-fwd.asc'
    M_p20 = 'SC_074-Mag-fwd.asc'
    M_n30 = 'SC_075-Mag-fwd.asc'
    
    #V_mag_dic = {0: M_org, 2: M_p2, -4: M_n4, 6: M_p6, -10: M_n10, 20: M_p20, -30: M_n30}
    V_mag_dic = {2: M_p2, -4: M_n4, 6: M_p6, -10: M_n10, 20: M_p20, -30: M_n30}
    V_vec = V_mag_dic.keys()    
    
    
    # create temporary mag objects. We will store the differences here
    mag_obj_diff = da.getMag(M_org)
    comb = []
    V_diff = []
    FWHM_vec = []
    
    # get combinations
    comb = []
    for subset in itertools.combinations(V_vec, 2):
        subset_ord = order(subset)
        comb.append(order(subset_ord))
        V_diff.append(subset_ord[0] - subset_ord[1])

    
    counter = len(comb)
    for subset in comb: 
        print(str(counter) + ' left')
        
        Vpos, Vneg = subset
        
        mat_diff, x_best, y_best = cq.chi_sqr(V_mag_dic[Vpos], V_mag_dic[Vneg], 30)
        mag_obj_diff.data = mat_diff * PHI0_TO_MPHI0
        
        mag_obj_diff.aveLineSubtract()  
            
        #mag_obj_diff.toString['title'] = '$\Delta V_{bg} = $' + str(V_diff[i]) + ' V'
        #mag_obj_diff.plotData(cbLabel= 'm$\Phi_0$', figNumber= i, CS= COLOR_SCALE)
    
        histReturn = mag_obj_diff.getHist()  
        (popt, pcov) = da.fitGauss(histReturn[0], histReturn[1], [.1, .1, .1])   
        FWHM_vec.append(popt[2])
        
        #plotting histogram
        #plt.plot(histReturn[0],da.funcGauss(histReturn[0],*popt),'r-')
        #plt.xlabel('m$\Phi_0$')
        counter -= 1
    
    plt.close('all')
    fig = plt.figure(100, facecolor = 'white')
    plt.plot(V_diff, FWHM_vec, 'bo')
    plt.xlabel('$\Delta V_{bg}$ (V)', fontsize = 16)
    plt.ylabel('$\sigma_{Mag}$ (m$\Phi_0$)', fontsize = 16)
    plt.title('Standard deviation of the magnetometry images difference between $V_{bg}$')
    return

def mean_sub():
    M_org = 'SC_069-Mag-fwd.asc'
    M_p2 = 'SC_070-Mag-fwd.asc'
    M_n4 = 'SC_071-Mag-fwd.asc'
    M_p6 = 'SC_072-Mag-fwd.asc'
    M_n10 = 'SC_073-Mag-fwd.asc'
    M_p20 = 'SC_074-Mag-fwd.asc'
    M_n30 = 'SC_075-Mag-fwd.asc'
    
    M_vec = [M_p2, M_n4, M_p6, M_n10, M_p20, M_n30]    
    V_vec = [2, -4, 6, -10, 20, -30]
    M_mean = []
    
    for M in M_vec:
        mag_obj = da.getData(M)
        M_mean.append(np.mean(mag_obj.data))

    M_mean = np.array(M_mean) 
    M_mean = (M_mean - M_mean[0]) * PHI0_TO_MPHI0
    V_vec = np.array(V_vec)

    plt.close('all')
    fig = plt.figure(100, facecolor = 'white')
    plt.plot(V_vec, M_mean, 'bo')
    plt.xlabel('$V_{bg}$ (V)', fontsize = 16)
    plt.ylabel('Normalized Average Mag (m$\Phi_0$)', fontsize = 16)
    return 
        
 
def main():
    gaussian_fits()
    #mean_sub()
   
if __name__ == '__main__':
    main()
    


    







