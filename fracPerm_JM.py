# A code to calculate the permeabilty of a fracture.
# The code assume Hagen-Poiseulle flow everywhere
# Carl Fredrik Berg, carl.f.berg@ntnu.no, 2025

# Modified by Jess McBeck

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg 
import time


# Import fracture widths
#csv_path = 'testFrac/test_aperture.csv'
# data made in get_aperture_csv.m
# x and y indexes (voxels), aperture in micrometers
#csv_path = 'C:/Users/jessicmc/research_local/aperture/txt/WG18_ap_grid_sc3.txt'
csv_path = 'C:/Users/jessicmc/research_local/aperture/txt/WG18_ap_grid_syn.txt'


if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# Load numeric values from CSV (handles optional header / irregular rows)
fracture_widths = np.loadtxt(csv_path, delimiter=',')

dim=[int(np.max(fracture_widths[:,0]))+1, int(np.max(fracture_widths[:,1]))+1]
#dim = [541, 541]
print("Dimensions of the fracture network:", dim)


# what is the point of this line?
#dim[0]=dim[0]

b=fracture_widths[:(dim[0])*(dim[1]),-1].reshape(int(dim[0]), int(dim[1]))
plt.imshow(b,cmap='jet')
plt.colorbar(label='geometric aperture, b')
plt.show()

conductanceGrid=b**3/(12.0)

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
                # is the leftmost boundary always an inlet?
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
xvec = scipy.sparse.linalg.cg(mtx, bvec, x0vec)
# print(xvec)
toc = time.time()
print('Time to solve the system:', toc-tic)

# Plot pressure field
pressureField=np.zeros((dim[0], dim[1]), dtype=float)
for jj in range(0, dim[1]):
    for ii in range(0, dim[0]):
        if conductanceGrid[ii][jj]>0:
            idx=gridc[ii][jj]
            pressureField[ii][jj]=xvec[0][idx]
        else:
            pressureField[ii][jj]=0
            
            # previous script: makes the totFlow=0
            #pressureField[ii][jj]=np.nan

plt.imshow(pressureField,cmap='jet')
plt.colorbar(label='Pressure')
plt.title('Pressure field in fracture network')
# plt.show()

# Calculate flow in cross-section
# why is the aperture (conductanceGrid) multiplied by 2?
# due to division by 2 in above loop?
totFlow=0 
# for jj in range(0, dim[1]):
#     #totFlow+=(pin-pressureField[0][jj])*conductanceGrid[0][jj]*2


# modified from the oval script
for jj in range(0, dim[1]):
    delp= (pressureField[int(dim[1]/2)][jj]-pressureField[int(dim[1]/2)+1][jj])
    totFlow+=delp*conductanceGrid[0][jj]*2
    

# for ii in range(0, dim[0]):
#     delp= (pressureField[ii][int(dim[1]/2)]-pressureField[ii][int(dim[1]/2)+1])
#     totFlow+=delp*conductanceGrid[0][ii]*2
    
    #if pressureField[ii][int(dim[1]/2)+1]>0 and pressureField[ii][int(dim[1]/2)]>0:
        #totFlow+=(pressureField[ii][int(dim[1]/2)]-pressureField[ii][int(dim[1]/2)+1])
        

print(totFlow)

#vox_um = 6.5 # from voxel to micron
# should the units of the aperture be meters?
# the totFlow is nan, what should the pressure field be when the aperture=0?
effctiveFracWidth=(12*totFlow*dim[0]/(pin-pout))**(1/3)
print('Effective fracture width: ', effctiveFracWidth)
