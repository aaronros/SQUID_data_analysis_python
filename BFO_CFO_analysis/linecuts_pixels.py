'''
x-linecuts of 5 pixels in y 
Created: 2017-01-23

Desc: For a set of images, take a linecut for a large feature
and plot the line cuts for each gate votlage. Must consider the pixel
shifts between the images

'''

from chi_squared_mod import *


'''
Data files
'''
M_org = 'SC_069-Mag-fwd.asc'
M_p2 = 'SC_070-Mag-fwd.asc'
M_n4 = 'SC_071-Mag-fwd.asc'
M_p6 = 'SC_072-Mag-fwd.asc'
M_n10 = 'SC_073-Mag-fwd.asc'
M_p20 = 'SC_074-Mag-fwd.asc'
M_n30 = 'SC_075-Mag-fwd.asc'

M_vec = [M_org, M_p2, M_n4, M_p6, M_n10, M_p20, M_n30]
V_vec = [0, 2, -4, 6, -10, 20, -30]   


'''
create this object so I have all the methods, 
but replace the obj.data secti
'''

mag_tmp_shift = getMag(M_vec[0]) 
mag_tmp_diff = getMag(M_vec[0]) 

'''
TODO!!

get specs for I, J. Create new shifted matrix
for n_best, m_best. Put that in the data of mag_tmp_shift
Take linecuts
'''

I = int(mag_tmp_shift.specs['xpixel'])
J = int(mag_tmp_shift.specs['ypixel'])

for i in range(1,len(M_vec)):
    
        
    (diff, x_best, y_best) = chi_sqr(M_vec[1], M_vec[i])
    mag_tmp_shift = getMag(M_vec[0])    
    M_shift = shiftMatrixHorz(mag_tmp_shift.data,I,J,x_best)
    M_shift = shiftMatrixVert(M_shift,I,J,y_best)  
    M_shift = M_shift[10:I-10, 10:J-10]                                     
    
    mag_tmp_shift.data = M_shift
    mag_tmp_shift.toString['title'] = 'Shifted image of' + M_vec[i]
    
    fig = figure(i, facecolor = 'white')
    
    linecut1 = mag_tmp_shift.horzLineCut(80-2)
    linecut2 = mag_tmp_shift.horzLineCut(80-1)
    linecut3 = mag_tmp_shift.horzLineCut(80+0)
    linecut4 = mag_tmp_shift.horzLineCut(80+1)
    linecut5 = mag_tmp_shift.horzLineCut(80+2)
    
    p1 = plot(linecut1[:,0], 1000*linecut1[:,1],label = '-2 pixels')
    p2 = plot(linecut2[:,0], 1000*linecut2[:,1],label = '-1 pixels')
    p3 = plot(linecut3[:,0], 1000*linecut3[:,1],label = '0 pixels')
    p4 = plot(linecut4[:,0], 1000*linecut4[:,1],label = '+1 pixels')
    p5 = plot(linecut5[:,0], 1000*linecut5[:,1],label = '+2 pixels')
    
    legend(['-2 pixels','-1 pixels', '0 pixels', '+1 pixels', '+2 pixels'])
    xlabel('x [$\mu$ m]', fontsize = 20)
    ylabel('m$\Phi_0$', fontsize = 20)
    title('$V_{bg}$ = ' + str(V_vec[i]), fontsize = 20)

#mag_tmp_shift.plotData()



#mag_tmp.data = diff    





