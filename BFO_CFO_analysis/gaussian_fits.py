'''
Gaussian fits of difference images
Created: 2017-01-23

Desc: For the set of difference images, take an average line subtraction
and plot the FWHM of each gaussian vs gate voltage

'''

from chi_squared_mod import *
import itertools

'''
Order such that the larger V comes first
'''
def order(subset, V_vec):
    if V_vec[subset[0]] < V_vec[subset[1]]:
        return (subset[1], subset[0])
    return (subset[0], subset[1])

M_org = 'SC_069-Mag-fwd.asc'
M_p2 = 'SC_070-Mag-fwd.asc'
M_n4 = 'SC_071-Mag-fwd.asc'
M_p6 = 'SC_072-Mag-fwd.asc'
M_n10 = 'SC_073-Mag-fwd.asc'
M_p20 = 'SC_074-Mag-fwd.asc'
M_n30 = 'SC_075-Mag-fwd.asc'

M_vec = [M_p2, M_n4, M_p6, M_n10, M_p20, M_n30]
V_vec = [2, -4, 6, -10, 20, -30]   
count_vec = [0,1,2,3,4,5]

# get combinations
comb = []
for subset in itertools.combinations(count_vec, 2):
    comb.append(subset)
'''
create this object so I have all the methods, 
but replace the obj.data secti
'''

mag_tmp_diff = getMag(M_vec[0]) 
mag_tmp_diff.toString['title'] = 'Difference Image'

'''
Now create difference and record FWHM for each subset
'''

V_diff_vec = []
FWHM_vec= []
    
for subset in comb:
    print(subset)
    subset_cor_order = order(subset,V_vec)
    M_diff, n_best, m_best = chi_sqr(M_vec[subset_cor_order[0]], M_vec[subset_cor_order[1]])
    
    V_diff_vec.append(V_vec[subset_cor_order[0]] - V_vec[subset_cor_order[1]])

    # put data in mag object
    mag_tmp_diff.data = M_diff 
    mag_tmp_diff.aveLineSubtract()
    
    # get hist
    histReturn = mag_tmp_diff.getHist()    
    (popt, pcov) = fitGauss(histReturn[0], histReturn[1], [.1, .1, .1])    
    
    FWHM_vec.append(popt[2])
plt.close('all')

fig = plt.figure(2, facecolor = 'white')
plt.plot(np.array(V_diff_vec), 1000*np.array(FWHM_vec), 'bo')
plt.xlabel('$\Delta V_{bg}$', fontsize = 16)
plt.ylabel('$\sigma_{Mag}$', fontsize = 16)
plt.title('Standard deviation of the magnetometry images difference between $V_{bg}$')



plt.plot(histReturn[0],funcGauss(histReturn[0],*popt),'r-')
#show()