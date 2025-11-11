# A code to calculate the permeabilty of a fracture.
# The code assume Hagen-Poiseulle flow everywhere
# Carl Fredrik Berg, carl.f.berg@ntnu.no, 2025

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg 
import time


# Import fracture widths
csv_path = 'testFrac/test_aperture.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

# Load numeric values from CSV (handles optional header / irregular rows)
fracture_widths = np.loadtxt(csv_path, delimiter=',')

dim=[int(np.max(fracture_widths[:,0])),int(np.max(fracture_widths[:,1]))]
print("Dimensions of the fracture network:", dim)

#print(np.max(fracture_widths[:,0]),np.max(fracture_widths[:,1]))
dim[0]=dim[0]

conductanceGrid=fracture_widths[:(dim[0])*(dim[1]),-1].reshape(int(dim[0]),int(dim[1]))
# plt.imshow(conductanceGrid,cmap='jet')
# plt.show()

# Assuming Hagen-Pouiseulle
conductanceGrid=conductanceGrid**3/(12.0)

# Cut out a smaller grid to test code
conductanceGrid=conductanceGrid[0:300,0:300]
dim=[300,300]
print("Dimensions of the fracture network:", dim)
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
            centCond=0
            # Left node
            if ii>0:
                if conductanceGrid[ii-1][jj]>0:
                    colc=gridc[ii-1][jj]
                    rowl.append(rowc)
                    coll.append(colc)
                    locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii-1][jj]**-1)
                    datal.append(locCond)
                    centCond-=locCond
            else:
                bvec[rowc]-=conductanceGrid[ii][jj]*2*pin #Multiply by 2 since the distance to the border is halved
                centCond-=conductanceGrid[ii][jj]*2
                nninlet+=1
            # Right node
            if ii<dim[0]-1:
                if conductanceGrid[ii+1][jj]>0:
                    colc=gridc[ii+1][jj]
                    rowl.append(rowc)
                    coll.append(colc)
                    locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii+1][jj]**-1)
                    datal.append(locCond)
                    centCond+=-locCond
            else:
                bvec[rowc]-=conductanceGrid[ii][jj]*2*pout
                centCond-=conductanceGrid[ii][jj]*2

            # Bottom node
            if jj>0:
                if conductanceGrid[ii][jj-1]>0:
                    colc=gridc[ii][jj-1]
                    rowl.append(rowc)
                    coll.append(colc)
                    locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii][jj-1]**-1)
                    datal.append(locCond)
                    centCond+=-locCond
            # Top node
            if jj<dim[1]-1:
                if conductanceGrid[ii][jj+1]>0:
                    colc=gridc[ii][jj+1]
                    rowl.append(rowc)
                    coll.append(colc)
                    locCond=2/(conductanceGrid[ii][jj]**-1+conductanceGrid[ii][jj+1]**-1)
                    datal.append(locCond)
                    centCond+=-locCond
            # Center node
            rowl.append(rowc)
            coll.append(rowc)
            datal.append(centCond)

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
plt.imshow(pressureField,cmap='jet')
plt.colorbar(label='Pressure')
plt.title('Pressure field in fracture network')
# plt.show()

# Calculate flow in cross-section
totFlow=0
for jj in range(0,dim[1]):
    if (pin-pressureField[0][jj])*conductanceGrid[0][jj]*2>0:
        totFlow+=(pin-pressureField[0][jj])*conductanceGrid[0][jj]*2

print(totFlow)

effctiveFracWidth=(12*totFlow*dim[0]/((pin-pout)*dim[1]))**(1/3)
print('Effective fracture width: ', effctiveFracWidth)
