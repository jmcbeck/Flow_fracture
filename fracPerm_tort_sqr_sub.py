# A code to calculate the permeabilty of a fracture.
# The code assume Hagen-Poiseulle flow everywhere
# Carl Fredrik Berg, carl.f.berg@ntnu.no, 2025

# modified for experimental data with a square dimension

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg 
import time
import statistics
import math

# Import fracture widths
#csv_path = 'testFrac/test_aperture.csv'
#csv_path = 'C:/Users/jessicmc/research_local/aperture/txt/WG18_ap_sqr_syn.txt'


# data made in the matlab file get_aperture_sqr_csv.m
# 3, 4, 5, 6, 7, 8, 9, 40, 65, 125,
#scanns = ['3', '4', '5', '6', '7', '8', '9', '40', '65', '125']
#scanns = ['15', '20', '25', '30', '35']
#scanns = ['3', '125']
#scanns = ['70', '75', '80', '85', '90', '95', '100', '105', '110', '115', '120', '121', '122', '123', '124']
#scanns = ['91', '92', '93', '94', '96', '97', '98', '99', '101', '102', '103', '104', '106', '107', '108', '109', '111', '112', '113', '114']

# spc = '100'
# gmax = 9
#scanns = ['3']
# spc = '50'
# gmax = 49
# spc = '375'
# gmax = 1

#spc = '25'
#gmax = 225
#scanns = ['3', '4', '5', '106', '107', '108', '123', '124', '125']

spc = '379'
gmax = 1
scanns = ['3', '4', '5', '6', '7', '8', '9', '15', '20', '25', '30', '35', '40', '65', '70', '75', '80', '85', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '120', '121', '122', '123', '124', '125']
res_file = 'C:/Users/jessicmc/research_local/aperture/python/output/results_flowtort_sqr_sub'+spc+'.txt'


for scann in scanns:

    gi=1
    while gi<=gmax:

        csv_path = 'C:/Users/jessicmc/research_local/aperture/txt/WG18_ap_sqr_sc'+scann+'_spc'+spc+'_g'+str(gi)+'.txt'
        
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load numeric values from CSV (handles optional header / irregular rows)
        fracture_widths = np.loadtxt(csv_path, delimiter=',')
        
        #dim=[int(np.max(fracture_widths[:,0])),int(np.max(fracture_widths[:,1]))]
        dim=[int(np.max(fracture_widths[:,0]))+1, int(np.max(fracture_widths[:,1]))+1]
        
        #print("Dimensions of the fracture network:", dim)
        
        
        b = fracture_widths[:(dim[0])*(dim[1]),-1].reshape(int(dim[0]), int(dim[1]))
        bgeomu = statistics.mean(fracture_widths[:,-1])
        bgeostd = statistics.stdev(fracture_widths[:,-1])
        
        conductanceGrid = (b**3)/(12.0)
        
        kn = np.zeros((dim[0],dim[1]),dtype=float)
        kmax = max(max(row) for row in conductanceGrid)
        for jj in range(0,dim[1]):
            for ii in range(0,dim[0]):
                kc = conductanceGrid[ii][jj]
                kn[ii][jj] = kc/kmax


        # plt.imshow(conductanceGrid, cmap='jet', vmin=0, vmax=100000)
        # plt.colorbar(label='k grid')
        # plt.show()
        
        # plt.imshow(kn, cmap='jet', vmin=0.0, vmax=0.25)
        # plt.colorbar(label='norm conductance Grid')
        # plt.show()
        

        
        # Cut out a smaller grid to test code
        # conductanceGrid=conductanceGrid[0:20,0:20]
        # dim=[20,20]
        # print("Dimensions of the fracture network:", dim)
        # plt.imshow(conductanceGrid,cmap='jet')
        # plt.show()
        
        
        gridc=np.zeros(dim,int)
        count=0
        for jj in range(0,dim[1]):
            for ii in range(0,dim[0]):
                if conductanceGrid[ii][jj]>0:
                    gridc[ii][jj]=count
                    count+=1
        
        bvec=np.zeros([count],dtype=float)
        x0vec=np.zeros([count],dtype=float)
        
        nninlet=0
        rowl=[]
        coll=[]
        datal=[]
        
        pin=1
        pout=0
        
        for jj in range(0,dim[1]):
            for ii in range(0,dim[0]):
                if conductanceGrid[ii][jj]>0:
                    rowc=gridc[ii][jj]
                    x0vec[rowc]=(pin-pout)*(dim[0]-float(ii))/float(dim[0]+1)
                    centint=0
                    # Left node
                    if ii>0:
                        if conductanceGrid[ii-1][jj]>0:
                            colc=gridc[ii-1][jj]
                            rowl.append(rowc)
                            coll.append(colc)
                            locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii-1][jj]**-1)
                            datal.append(locCond)
                            centint-=locCond
                    else:
                        bvec[rowc]-=conductanceGrid[ii][jj]*2*pin #Multiply by 2 since the distance to the border is halved
                        centint-=conductanceGrid[ii][jj]*2
                        nninlet+=1
                    # Right node
                    if ii<dim[0]-1:
                        if conductanceGrid[ii+1][jj]>0:
                            colc=gridc[ii+1][jj]
                            rowl.append(rowc)
                            coll.append(colc)
                            locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii+1][jj]**-1)
                            datal.append(locCond)
                            centint+=-locCond
                    else:
                        bvec[rowc]-=conductanceGrid[ii][jj]*2*pout
                        centint-=conductanceGrid[ii][jj]*2
        
                    # Bottom node
                    if jj>0:
                        if conductanceGrid[ii][jj-1]>0:
                            colc=gridc[ii][jj-1]
                            rowl.append(rowc)
                            coll.append(colc)
                            locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii][jj-1]**-1)
                            datal.append(locCond)
                            centint+=-locCond
                    # Top node
                    if jj<dim[1]-1:
                        if conductanceGrid[ii][jj+1]>0:
                            colc=gridc[ii][jj+1]
                            rowl.append(rowc)
                            coll.append(colc)
                            locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii][jj+1]**-1)
                            datal.append(locCond)
                            centint+=-locCond
                    # Center node
                    rowl.append(rowc)
                    coll.append(rowc)
                    datal.append(centint)
        
        row=np.array(rowl)
        col=np.array(coll)
        data=np.array(datal)
        
        # Create sparse matrix and solve
        tic= time.time()
        mtx=sparse.csc_matrix((data, (col, row)), shape=(count,count),dtype=float)
        # print(mtx)
        # print(bvec)
        # print(x0vec)
        xvec = scipy.sparse.linalg.cg(mtx,bvec,x0vec)
        # print(xvec)
        toc = time.time()
        print('Time to solve the system:', toc-tic)
        
        # Plot pressure field
        pressureField=np.zeros((dim[0],dim[1]),dtype=float)
        for jj in range(0,dim[1]):
            for ii in range(0,dim[0]):
                if conductanceGrid[ii][jj]>0:
                    idx=gridc[ii][jj]
                    pressureField[ii][jj]=xvec[0][idx]
                else:
                    pressureField[ii][jj]=np.nan
        
        # plt.imshow(pressureField,cmap='jet')
        # plt.colorbar(label='Pressure')
        # plt.title('Pressure field in fracture network')
        # plt.show()
 
        
        # Calculate flow in cross-section at the top boundary
        totFlow=0
        for jj in range(0,dim[1]):
            if math.isnan(pressureField[0][jj])==0:
                totFlow+=(pin-pressureField[0][jj])*conductanceGrid[0][jj]*2
        
        #print(totFlow)
        # gradient over rows (y-direction), and columns (x-direction)
        # the flow rate is the pressure field* the conductance * 2
        
        print('Calculating the flow field')
        
        flow = np.zeros((dim[0],dim[1]), dtype=float)
        for jj in range(0, dim[1]):
            for ii in range(0, dim[0]):
                pc = pressureField[ii][jj]
                delpc = pin-pc
                kc = conductanceGrid[ii][jj]*2
                flow[ii][jj] = delpc*kc
                
                
        fy, fx = np.gradient(flow)
        py, px = np.gradient(pressureField)
        
        # plt.title('flow')
        # plt.imshow(flow, cmap='hot', vmin=0, vmax=100000)
        # plt.colorbar()
        # plt.show()
        
        # plt.title('gradient of flow, fy')
        # plt.imshow(fy, cmap='seismic', vmin=-1000, vmax=1000)
        # plt.colorbar()
        # plt.show()
        
        # plt.title('gradient of flow, fx')
        # plt.imshow(fx, cmap='seismic', vmin=-1000, vmax=1000)
        # plt.colorbar()
        # plt.show()
       
        
        # for each column
        # attempting to calculate tortuosity following Brown 1987
        x = list(range(0, dim[0]))
        ftlocv = []
        ftlocnv = []
        ftlocs = [[math.nan for _ in range(dim[0])] for _ in range(dim[1])]

        ptlocv = []
        ptlocnv = []
        ptlocs = [[math.nan for _ in range(dim[0])] for _ in range(dim[1])]


        print('Getting the tortuosity')
        for jj in range(0, dim[1]):
            for ii in range(0, dim[0]):
                #pxc = abs(px[ii][jj])
                fyc = abs(fy[ii][jj])
                fmag = math.sqrt(fyc**2 + fx[ii][jj]**2)
                
                pyc = abs(py[ii][jj])
                pmag = math.sqrt(pyc**2 + px[ii][jj]**2)

                # weight this value by the local flow rate, the fraction of this value to the maximum
                
                if math.isnan(fyc)==0 and math.isnan(fmag)==0:
                    wn = fmag
                    ftloc = (fmag/fyc)
                    ftlocv.append(ftloc)
                    nv = int(wn/10)
                    #print(nv)
                    if nv>0:
                        for _ in range(nv):
                            ftlocnv.append(ftloc)
                            
                    wn = kn[ii][jj]
                    ptloc = (pmag/pyc)
                    ptlocv.append(ptloc)
                    nv = int(wn*1000)
                    #print(wn, nv)
                    if nv>0:
                        for _ in range(nv):
                            ptlocnv.append(ptloc)

                else:
                    ftloc = math.nan
                    ptloc = math.nan
                    
                ftlocs[ii][jj] = ftloc
                ptlocs[ii][jj] = ptloc


            
         
        ftlocav = sum(ftlocv)/len(ftlocv)
        ftlocnav = sum(ftlocnv)/len(ftlocnv)
        
        ptlocav = sum(ptlocv)/len(ptlocv)
        ptlocnav = sum(ptlocnv)/len(ptlocnv)
        
        # plt.title('local tortuosity from pressure field')
        # plt.imshow(ptlocs, cmap='hot', vmin=1, vmax=50)
        # plt.colorbar()
        # plt.show()
        
        
     
        # removed dim[0] in the equation below
        # and then the effective fracture width is close to 1 for synthetic data
        #bhyd=(12*totFlow*dim[0]/(pin-pout))**(1/3)
        bhyd=((12*totFlow)/(pin-pout))**(1/3)
        
        
        #print('Effective fracture width: ', bhyd)
        
        print('Scan #, g, b_geo mean, std, b_hyd, tortF, weighted tortF, tortP, weighted tortP,')
        new_res = scann+' '+str(gi)+' '+str(bgeomu)+' '+str(bgeostd)+' '+str(bhyd)+' '+str(ftlocav)+' '+str(ftlocnav)+' '+str(ptlocav)+' '+str(ptlocnav)+'\n'
        print(new_res)
        gi=gi+1


        with open(res_file, 'a') as file:
            # Write the string to the file
            file.write(new_res)
        
