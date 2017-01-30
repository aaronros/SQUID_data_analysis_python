'''
Chi^2 routine. Takes two similar matrices and finds the optimal
displacement by looking at Chi^2
X^2 = 1/(IJ) * sum(i,j) (M(i,j) - M(i+n,j+m))^2

'''


'''
boilerplate imports and global constants initialization
'''

from data_analysis_module import *

set_VtoPhi(0.332)
set_GainDC(10)

plt.close('all')

# Part 1: declare function
# ---------------------------------------------------------
# shift matrix by some pixel. Can be simiplied to 1 funtion
# but I got lazy

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

    # Part 2: Initialize Matrices and other related parameters    
    # --------------------------------------------------------

def chi_sqr(asc1, asc2, sub):

    mag1 = getMag(asc1)
    #mag1.plotData(figNumber = 1)
    #title('Vgate = +0V at 4K',fontsize = 16)
    
    mag2 = getMag(asc2)
    #mag2.plotData(figNumber = 2)
    #title('Vgate = -30V applied at 150K',fontsize = 16)
    
    I = int(mag1.specs['xpixel'])
    J = int(mag1.specs['ypixel'])
    
    if (mag1.specs ['xpixel'] != mag2.specs ['xpixel']) | (mag1.specs ['ypixel'] != mag2.specs ['ypixel']):
        print('Warning: x and y pixels are not the same between matrices')
    
        # Part 3: Perform X^2 routine
        # --------------------------------------------------
        # define how much to shift the matrix by in shift_pixels
        # can be different for x and y, but for this code, we will 
        # shift the matrices equal amounts
    
    shift_pixels = np.linspace(-5,5,11)
    
    X_sqr = np.zeros([shift_pixels.size,shift_pixels.size])
    for n in range(shift_pixels.size):
        #print 'n = ' + str(shift_pixels[n])
        for m in range(shift_pixels.size):
            X_sqr_nm = 0
            M1 = mag1.data
            M2 = shiftMatrixHorz(mag2.data,I,J,shift_pixels[n])
            M2 = shiftMatrixVert(M2,I,J,shift_pixels[m])  
            for i in range(10,I-10): # only look at values not wrapped
                for j in range(10,J-10): # only look at values not wrapped
                    X_sqr_temp = (M1[i,j] - M2[i,j]) * (M1[i,j] - M2[i,j])
                    X_sqr_nm = X_sqr_nm + X_sqr_temp                                       
            X_sqr[n,m] = X_sqr_nm
    
        # Part 4: Determine best n and m that minimizes X^2
        # -------------------------------------------------
    
    temp = np.where(X_sqr == X_sqr.min())
    n_best = shift_pixels[temp[0][0]]
    m_best = shift_pixels[temp[1][0]]
    
        # Part 5: Plot all the things
        # -------------------------------------------------
    
            
    
    
    # plot Matrix1 - shifted(Matrix2) for best shifted value determined by X^2
#    fig2 = figure(5, facecolor = 'white')
    M1 = mag1.data
    M2 = shiftMatrixHorz(mag2.data,I,J,n_best)
    M2 = shiftMatrixVert(M2,I,J,m_best)  
    
    M3 = M1 - M2
    M3new = np.zeros((I - sub,J - sub))
    for i in range(I - sub): # only look at values not wrapped
        for j in range(J - sub): # only look at values not wrapped
            M3new[i,j] = M3[i + sub,j + sub]
    
#    
#    imshow(M3new,'RdYlGn')
#    cb = colorbar(orientation = 'vertical')
    #clim(-0.01,0.01) 
#    title('Poled Magneomtry Image Subtracted for Unpoled Image',fontsize = 16)
#    show()
            
    Mat_diff = M3new
    x_best = n_best
    y_best = m_best

    return Mat_diff, x_best, y_best
      
        
        
        