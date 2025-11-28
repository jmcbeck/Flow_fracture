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
res_file = 'C:/Users/jessicmc/research_local/aperture/python/output/results_qtort_sqr_sub'+spc+'.txt'


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
        
     


        plt.imshow(b, cmap='jet')
        plt.colorbar(label='aperture')
        plt.show()
        
    

        
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
        flowm = np.zeros((dim[0],dim[1]), dtype=float)
        flowv = np.zeros((dim[0],dim[1]), dtype=float)
        flowu = np.zeros((dim[0],dim[1]), dtype=float)
        flowd = np.zeros((dim[0],dim[1]), dtype=float)
        flowl = np.zeros((dim[0],dim[1]), dtype=float)
        flowr = np.zeros((dim[0],dim[1]), dtype=float)
        
        tortg = np.zeros((dim[0],dim[1]), dtype=float)
        
        tort = []
        tortw = []
        
        # py, px = np.gradient(pressureField) #This gives the pressure gradients
        # effCondX=2/(1/conductanceGrid[:-1][:]+1/conductanceGrid[1:][:])
        # effCondY=2/(1/conductanceGrid[:][:-1]+1/conductanceGrid[:][1:])
        
        
        for jj in range(1, dim[1]-1):
            jc1 = jj-1
            jc2 = jj+1
            for ii in range(1, dim[0]-1):
                ic1 = ii-1
                ic2 = ii+1

                pc = pressureField[ii][jj]
                kc = conductanceGrid[ii][jj]
                
                pl = pressureField[ii][jc1]
                pr = pressureField[ii][jc2]
                
                pu = pressureField[ic1][jj]
                pd = pressureField[ic2][jj]
                
                kl = conductanceGrid[ii][jc1]
                kr = conductanceGrid[ii][jc2]
                
                ku = conductanceGrid[ic1][jj]
                kd = conductanceGrid[ic2][jj]
                
                if kl>0 and kr>0 and ku>0 and kd>0 and kc>0:
                
                    kel = 2/(1/kl + 1/kc)
                    ker = 2/(1/kr + 1/kc)
    
                    keu = 2/(1/ku + 1/kc)
                    ked = 2/(1/kd + 1/kc)
                    
                    dpl = pl-pc
                    dpr = pr-pc
                    
                    dpu = pu-pc
                    dpd = pd-pc
                    
                    ql = dpl*kel
                    qr = dpr*ker
                    qu = dpu*keu
                    qd = dpd*ked
                    
                    qc = statistics.mean([ql, qr, qu, qd])
                    
                    qv = statistics.mean([qu, qd])
                    qm = math.sqrt(qv**2 + statistics.mean([ql, qr])**2)
                    
                    flowm[ii][jj] = qm
                    flow[ii][jj] = qc
                    flowu[ii][jj] = qu
                    flowd[ii][jj] = qd
                    flowv[ii][jj] = qv
                    flowl[ii][jj] = ql
                    flowr[ii][jj] = qr
                    
                    # tortuosity is average flow/flowd?
                    # or magnitude of flow/flow down
                    tc = abs(qm/qv)
                    tort.append(tc)
                    
                    wn = abs(qm)
                    nv = int(wn*100)
                    if nv>0:
                        for _ in range(nv):
                            tortw.append(tc)
                    
                    tortg[ii][jj] = tc
                
                #delpc = pin-pc
                #kc = conductanceGrid[ii][jj]*2
                #flow[ii][jj] = delpc*kc
                
        
        # plt.title('flow average')
        # plt.imshow(flow, cmap='hot', vmin=-0.1, vmax=0.1)
        # plt.colorbar()
        # plt.show()
        
        # plt.title('flow magnitude')
        # plt.imshow(flowm, cmap='hot', vmin=0, vmax=10)
        # plt.colorbar()
        # plt.show()
        
        # plt.title('flow d')
        # plt.imshow(flowv, cmap='hot', vmin=-5, vmax=5)
        # plt.colorbar()
        # plt.show()
        
        # plt.title('tortuosity=magnitude/average vertical')
        # plt.imshow(tortg, cmap='hot', vmin=1, vmax=2)
        # plt.colorbar()
        # plt.show()

       
    
        tortav = sum(tort)/len(tort)
        tortavw = sum(tortw)/len(tortw)

    
        # plt.title('local tortuosity from pressure field')
        # plt.imshow(ptlocs, cmap='hot', vmin=1, vmax=50)
        # plt.colorbar()
        # plt.show()
        
        
     
        # removed dim[0] in the equation below
        # and then the effective fracture width is close to 1 for synthetic data
        #bhyd=(12*totFlow*dim[0]/(pin-pout))**(1/3)
        #bhyd=((12*totFlow)/(pin-pout))**(1/3)
        
        totFn = totFlow/dim[1]
        delpn = (pin-pout)/dim[0]
        bhyd=((12*totFn)/delpn)**(1/3)
        
        #abc
        #print('Effective fracture width: ', bhyd)
        
        print('Scan #, g, b_geo mean, std, b_hyd, tort, weighted tort')
        new_res = scann+' '+str(gi)+' '+str(bgeomu)+' '+str(bgeostd)+' '+str(bhyd)+' '+str(tortav)+' '+str(tortavw)+'\n'
        print(new_res)
        gi=gi+1

        with open(res_file, 'a') as file:
            # Write the string to the file
            file.write(new_res)
        
