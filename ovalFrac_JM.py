# A code to calculate the hydraulic conductance of a fracture.
# The code assume Hagen-Poiseulle flow everywhere
# Carl Fredrik Berg, carl.f.berg@ntnu.no, 2025

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg 
import time

# -94691.9999999998	-96387.4730021599	-96640.5405405407	-96533.3333333335	-96250.0000000000

#pin = [-94691.9999999998	-96387.4730021599	-96640.5405405407	-96533.3333333335,	-96250.0000000000]

#pin = 94691 #Pa
#pout = 1
pin = 2E5 # Pascal
pout = 1E5 # Pascal

#QRate=1E-6 # m3/s|
QRate = 7.88415897435902E-10 # m3/s

def generate_oval_grid(nx, ny,
                      dim1_mm,
                      dim2_mm, 
                      b_dia, 
                      ybwall):
    """
    Create a grid with 1 inside an ellipse (dimensions in mm) and 0 outside.
    Returns a numpy array of shape (nx, ny).
    """
    a = dim1_mm / 2.0
    b = dim2_mm / 2.0

    # cell-center coordinates (x along first axis, y along second axis)
    x = np.linspace(-a, a, nx)
    y = np.linspace(-b, b, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')  # xv.shape == (nx, ny)

    # Main oval
    mask = (xv / a) ** 2 + (yv / b) ** 2 <= 1.0
    grid = np.where(mask, 1.0, 0.0)

    # Small oval at bottom (-1)
    a_small = 2.0 / 2.0
    
    b_small = b_dia / 2.0 # test2, fault at 30 deg
    
    # original
    #y_bottom = -b + 1.9 + 1.95 # why is it 1.95?
    y_bottom = -b + ybwall + ybwall
    mask_bottom = (xv / a_small) ** 2 + ((yv - y_bottom) / b_small) ** 2 <= 1.0
    grid = np.where(mask_bottom, -pin, grid)

    # Small oval at top (-2)
    y_top = b - (ybwall + ybwall)
    mask_top = (xv / a_small) ** 2 + ((yv - y_top) / b_small) ** 2 <= 1.0
    grid = np.where(mask_top, -pout, grid)

    return grid

dim1_mm = 38.0


testn = 4
if testn==1:
    # test #1, fault at 30 deg
    # radius of the borehole along the fault
    b_dia = 4.0  
    # distance between the boreholes on the fault
    L0 = 64.0
    # distance between borehole and wall along the fault
    ybwall = 2.0
    # slip
    #ux = 0
    #ux = 1
    ux = 3
    
elif testn==2:

    # test #2
    # fault at 31 deg
    # radius of the borehole along the fault
    b_dia = 3.9  
    # distance between the boreholes on the fault
    L0 = 62.1
    # distance between borehole and wall along the fault
    ybwall = 1.95
    # slip
    #ux = 0
    #ux = 1
    ux = 3
    
elif testn==3:
    # fault at 44.7 deg
    # test #3
    # radius of the borehole along the fault
    b_dia = 2.84  
    # distance between the boreholes on the fault
    L0 = 45.5
    # distance between borehole and wall along the fault
    ybwall = 1.42
    # slip
    #ux = 0
    #ux = 1
    ux = 3

else:

    # # fault at 46 deg
    # test #4
    # radius of the borehole along the fault
    b_dia = 2.8
    # distance between the boreholes on the fault
    L0 = 43.26
    # distance between borehole and wall along the fault
    ybwall = 2.1
    # slip
    #ux = 0
    #ux = 1
    ux = 3


dim2_mm = (L0-ux) + 2*(b_dia) + 2*(ybwall)



nx_cells = int(dim1_mm * 10)  # 0.1 mm resolution grid size= 1E-4
ny_cells = int(dim2_mm * 10)  # 0.1 mm resolution

dim=(nx_cells, ny_cells)
fractureGrid = generate_oval_grid(dim[0], dim[1], dim1_mm, dim2_mm, b_dia, ybwall)

#plt.imshow(fractureGrid,cmap='jet')
#plt.show()

gridc=np.zeros(dim,int)
count=0
for jj in range(0,dim[1]):
    for ii in range(0,dim[0]):
        if fractureGrid[ii][jj]>0:
            gridc[ii][jj]=count
            count+=1

bvec=np.zeros([count],dtype=float)
x0vec=np.zeros([count],dtype=float)

nninlet=0
rowl=[]
coll=[]
datal=[]


for jj in range(0,dim[1]):
    for ii in range(0,dim[0]):
        if fractureGrid[ii][jj]>0:
            rowc=gridc[ii][jj]
            x0vec[rowc]=(pin-pout)*(dim[0]-float(ii))/float(dim[0]+1)
            centCond=0
            # Left node
            if ii>0:
                if fractureGrid[ii-1][jj]>0:
                    colc=gridc[ii-1][jj]
                    rowl.append(rowc)
                    coll.append(colc)
                    datal.append(1)
                    centCond-=1
                if fractureGrid[ii-1][jj]<0:
                    bvec[rowc]+=2*fractureGrid[ii-1][jj] #Multiply by 2 since the distance to the border is halved
                    centCond-=2
            # Right node
            if ii<dim[0]-1:
                if fractureGrid[ii+1][jj]>0:
                    colc=gridc[ii+1][jj]
                    rowl.append(rowc)
                    coll.append(colc)
                    datal.append(1)
                    centCond+=-1
                if fractureGrid[ii+1][jj]<0:
                    bvec[rowc]+=2*fractureGrid[ii+1][jj]
                    centCond-=2
            # Bottom node
            if jj>0:
                if fractureGrid[ii][jj-1]>0:
                    colc=gridc[ii][jj-1]
                    rowl.append(rowc)
                    coll.append(colc)
                    datal.append(1)
                    centCond+=-1
                if fractureGrid[ii][jj-1]<0:
                    bvec[rowc]+=2*fractureGrid[ii][jj-1]
                    centCond-=2
            # Top node
            if jj<dim[1]-1:
                if fractureGrid[ii][jj+1]>0:
                    colc=gridc[ii][jj+1]
                    rowl.append(rowc)
                    coll.append(colc)
                    datal.append(1)
                    centCond+=-1
                if fractureGrid[ii][jj+1]<0:
                    bvec[rowc]+=2*fractureGrid[ii][jj+1]
                    centCond-=2
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

np.save('ovalFracPressureField.npy',xvec[0])

# Plot pressure field
pressureField=np.zeros((dim[0],dim[1]),dtype=float)
for jj in range(0,dim[1]):
    for ii in range(0,dim[0]):
        if fractureGrid[ii][jj]>0:
            idx=gridc[ii][jj]
            pressureField[ii][jj]=xvec[0][idx]
        else:
            pressureField[ii][jj]=np.nan
plt.imshow(pressureField,cmap='jet')
plt.colorbar(label='Pressure')
plt.title('Pressure field in fracture network')
plt.show()

# Calculate flow in cross-section
totFlow=0
for ii in range(0,dim[0]):
    if pressureField[ii][int(dim[1]/2)+1]>0 and pressureField[ii][int(dim[1]/2)]>0:
        totFlow+=(pressureField[ii][int(dim[1]/2)]-pressureField[ii][int(dim[1]/2)+1])
        #print(pressureField[int(dim[0]/2)][jj]-pressureField[int(dim[0]/2)+1][jj],pressureField[int(dim[0]/2)+1][jj],pressureField[int(dim[0]/2)][jj],totFlow)

print('Qapp=', totFlow)

relVelo=QRate/totFlow
fracAperture=(relVelo*12*1E-3*1E-4)**(1/3) # m (we are using the viscosity of water 1E-3 Pa s and a grid size of 1E-4 m)
print('Estimated average fracture aperture [m]: ', fracAperture)
