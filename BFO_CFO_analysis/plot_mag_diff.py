'''
Plot difference in magnetometry images

Created: 2017-01-24

Desc: Loads the raw magnetometry data and subtracts the different images

'''

import data_analysis_module as da
import chi_squared_mod as cq
import matplotlib.pyplot as plt

# global constnats
PHI0_TO_MPHI0 = 1000
COLOR_SCALE = [-1.5,1.5]
da.set_VtoPhi(0.332)
da.set_GainDC(10)

plt.close('all')

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
    V_mag_dic = {0: M_org, 2: M_p2, -4: M_n4, 6: M_p6, -10: M_n10, 20: M_p20, -30: M_n30}
    comb = [(2,-4), (6,-4), (6,-10), (20,-10), (20,-30)]
    V_diff = [6, 10, 16, 30, 50]
    
    # create temporary mag objects. We will store the differences here
    mag_obj_diff = da.getMag(M_org)
    FWHM_vec = []
    
    for i in range(len(comb)): 
        print(i)
        Vpos, Vneg = comb[i]
        
        mat_diff, x_best, y_best = cq.chi_sqr(V_mag_dic[Vpos], V_mag_dic[Vneg], 30)
        mag_obj_diff.data = mat_diff * PHI0_TO_MPHI0
        
        mag_obj_diff.aveLineSubtract()  
            
        mag_obj_diff.toString['title'] = '$\Delta V_{bg} = $' + str(V_diff[i]) + ' V'
        mag_obj_diff.plotData(cbLabel= 'm$\Phi_0$', figNumber= i, CS= COLOR_SCALE)
    
        histReturn = mag_obj_diff.getHist()  
        (popt, pcov) = da.fitGauss(histReturn[0], histReturn[1], [.1, .1, .1])   
        FWHM_vec.append(popt[2])
        
        #plotting histogram
        #plt.plot(histReturn[0],da.funcGauss(histReturn[0],*popt),'r-')
        #plt.xlabel('m$\Phi_0$')
    
    #plt.close('all')
    fig = plt.figure(200, facecolor = 'white')
    plt.plot(V_diff, FWHM_vec, 'bo')
    plt.xlabel('$\Delta V_{bg}$ (V)', fontsize = 16)
    plt.ylabel('$\sigma_{Mag}$ (m$\Phi_0$)', fontsize = 16)
    plt.title('Standard deviation of the magnetometry images difference between $V_{bg}$')
    return
    
if __name__ == '__main__':
    main()

    







