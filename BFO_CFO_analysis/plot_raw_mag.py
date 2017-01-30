'''
Plot raw magnetometry images

Created: 2017-01-24

Desc: Loads the raw magnetometry data and plots it
'''

import data_analysis_module as da
import matplotlib.pyplot as plt

# global constnats
PHI0_TO_MPHI0 = 1000
COLOR_SCALE = [-16,16]
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
    
    for i in range(len(M_vec)):    
        mag_obj = da.getMag(M_vec[i])
        mag_obj.data = mag_obj.data * PHI0_TO_MPHI0 
        
        mag_obj.toString['title'] = 'Magnetometry Scan after $V_{bg} = $' + str(V_vec[i]) + ' V'
        mag_obj.plotData(cbLabel= 'm$\Phi_0$', figNumber= i, CS= COLOR_SCALE)

    return
    
if __name__ == '__main__':
    main()







