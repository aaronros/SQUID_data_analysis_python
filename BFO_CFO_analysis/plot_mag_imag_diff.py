'''
Plot raw magnetometry images

Created: 2017-01-31

Desc: Loads the raw magnetometry data and plots it
'''

import sys
sys.path.append('../source/')
import data_analysis_module as da

import matplotlib.pyplot as plt

# global constants
PHI0_TO_MPHI0 = 1000
COLOR_SCALE = [-16,16]
da.set_VtoPhi(0.332)
da.set_GainDC(10)
PIXELS_SHIFT = 15

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
   
    mag_org_obj = da.getMag(M_org)
    mag_org_obj.data = mag_org_obj.data * PHI0_TO_MPHI0 
    mag_org_obj.toString['title'] = 'Magnetometry Scan after $V_{bg} = $' + str(V_vec[0]) + ' V'
    mag_org_obj.plotData(cbLabel= 'm$\Phi_0$', figNumber= 0, CS= COLOR_SCALE)
   
    for i in range(1, len(M_vec)):    
        mag_obj = da.getMag(M_vec[i])
        mag_obj.data = mag_obj.data * PHI0_TO_MPHI0 
        
        M_diff, x_best, y_best = da.align_scans(mag_org_obj, mag_obj, 
                                                PIXELS_SHIFT, plot_cor = False)
                                                
        mag_obj.data = da.shiftMatrixHorz(mag_obj.data, int(mag_obj.specs['xpixel']), 
                                          int(mag_obj.specs['ypixel']), y_best)
                                         
        mag_obj.data =  da.shiftMatrixHorz(mag_obj.data, int(mag_obj.specs['xpixel']), 
                                           int(mag_obj.specs['xpixel']), x_best) 
        
        mag_obj.data = mag_obj.data[50:150, 50:150]


        mag_obj.toString['title'] = 'Magnetometry Scan after $V_{bg} = $' + str(V_vec[i]) + ' V'
        mag_obj.plotData(cbLabel= 'm$\Phi_0$', figNumber= i, CS= COLOR_SCALE)

        if x_best == 0 and y_best == 0: 
            print('Warning: no shift found between ' + M_vec[i] + ' and ' + M_org)

    return
    
if __name__ == '__main__':
    main()







