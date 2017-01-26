'''
Create: 01-26-2017
Example of the data anaylsis module that loads scans, plots them
and then compares the two images
'''

import data_analysis_module as da

# constants
PHI0_TO_mPHI0 = 1000
CS_mag1 = [-16, 16]
CS_diff = [-2, 2]
PIXEL_SHIFT = 3

def main():
    # initialize measurements constants
    da.set_VtoPhi(0.332)
    da.set_GainDC(10)
    da.set_plot_font(18)
    filename1 ='SC_074-Mag-fwd.asc' 
    filename2 ='SC_075-Mag-fwd.asc' 
    
    # load magnetometry data
    mag1 = da.getMag(filename1)
    mag1.data *= PHI0_TO_mPHI0
    mag2 = da.getMag(filename2)
    mag2.data *= PHI0_TO_mPHI0
    
    # plot the first images
    mag1.toString['title'] = 'Magnetometry images at $V_{bg} = +20 V$'    
    mag1.plotData(cmap = 'RdYlGn', CS = CS_mag1, cbLabel = 'm$\Phi_0$', figNumber = 1)
        
    mag2.toString['title'] = 'Magnetometry images at $V_{bg} = -30 V$'
    mag2.plotData(cmap = 'RdYlGn', CS = CS_mag1, cbLabel = 'm$\Phi_0$', figNumber = 2)
    
    
    # find best pixel shift to align images
    mat_diff, x_best, y_best = da.align_scans(mag1, mag2, PIXEL_SHIFT)
    
    # plot the difference matrix
    mag1.data = mat_diff
    mag1.toString['title'] = 'Difference image'
    mag1.plotData(cmap = 'RdYlGn', CS = CS_diff, cbLabel = 'm$\Phi_0$',figNumber = 3)

    return

if __name__ == '__main__':
    main()

