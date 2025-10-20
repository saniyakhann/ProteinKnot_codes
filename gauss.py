##few corrections made from Djordje's original code to avoid NaN values and asymmetric matrices (all from slight computational errors)
import numpy as np
from numba import prange, njit

'''
Compute StS Writhe matrix between two rings or the same ring if ring1=ring2

arguments:
ring1 - np.array with xyz coordinates of the first ring (shape=(ncoord,3))
ring2 - np.array with xyz coordinates of the second ring (shape=(ncoord,3))
lw - window length for segment-to-segment writhe computation

'''
@njit
def compute_sts_writhe(ring1, ring2, lw):
    n1, n2 = ring1.shape[0], ring2.shape[0]  #added this line
    #matrix = np.zeros((ring1.shape[0],ring2.shape[0]))
    matrix = np.zeros((n1,n2))

   
    #instead of computing matrix[i,j] and matrix[j,i] independently (which is where the floating point difference could occur)
    #equate them to be the same computed valeu (writhe matrices should be symmetric anyways)
    #compute only upper triangle
    for i in prange(n1):
        for j in prange(i, n2): #start from i instead of 0 
            value = compute_single_sts_writhe(ring1, ring2, i, j, lw)
            matrix[i,j] = value 
            if i < j: #mirror to lower triangle
                matrix[j,i] = value #exact same value 
                
    return matrix

'''
Compute StS Writhe between two segments belonging to two rings or the same ring if ring1=ring2

arguments:
ring1 - np.array with xyz coordinates of the first ring (shape=(ncoord,3))
ring2 - np.array with xyz coordinates of the second ring (shape=(ncoord,3))
i - index of segment for the first ring
j - index of segment for the second ring
lw - window length for segment-to-segment writhe computation

'''
@njit
def compute_single_sts_writhe(ring1, ring2, i, j, lw):

    wr = 0
    epsilon = 1e-12
    
    # Loop over the segment of the first ring
    for it in prange(-np.int64(lw/2)+i,np.int64(lw/2)+i):
        # Loop over the segment of the second ring
        for jt in prange(-np.int64(lw/2)+j, np.int64(lw/2)+j): 
            
            one = ring1[np.mod(it-1,ring1.shape[0]),:]
            three = ring2[np.mod(jt-1,ring2.shape[0]),:]
            two = ring1[np.mod(it,ring1.shape[0]),:]
            four = ring2[np.mod(jt,ring2.shape[0]),:]

            r12=two-one
            r34=four-three
            r23=three-two
            r13=three-one
            r14=four-one
            r24=four-two

            n1 = np.cross(r13,r14)
            #if np.linalg.norm(n1)==0:
            n1_norm = np.linalg.norm(n1)
            if n1_norm < epsilon:
                continue
            n1 = n1 / n1_norm

            n2 = np.cross(r14,r24)
            #if np.linalg.norm(n2)==0:
            n2_norm = np.linalg.norm(n2)
            if n2_norm < epsilon:
                continue
            n2 = n2 / n2_norm

            n3 = np.cross(r24,r23)
            n3_norm = np.linalg.norm(n3)
            if n3_norm < epsilon:
                continue
            n3 = n3 / n3_norm

            n4 = np.cross(r23,r13)
            #if np.linalg.norm(n4)==0:
            n4_norm = np.linalg.norm(n4)
            if n4_norm < epsilon:
                continue
            n4 = n4 / n4_norm

            n1n2=np.dot(n1,n2)  
            n2n3=np.dot(n2,n3)  
            n3n4=np.dot(n3,n4)
            n4n1=np.dot(n4,n1)

            n1n2 = max(min(n1n2, 1.0 - epsilon), -1.0 + epsilon)  
            n2n3 = max(min(n2n3, 1.0 - epsilon), -1.0 + epsilon)
            n3n4 = max(min(n3n4, 1.0 - epsilon), -1.0 + epsilon)
            n4n1 = max(min(n4n1, 1.0 - epsilon), -1.0 + epsilon)

            cvec = np.cross(r34,r12)
            dprcvec = np.dot(cvec,r13)

            #if dprcvec == 0:
            if abs(dprcvec) < epsilon: #instead of the exact 0 check 
                continue

            #this causes NaN when input > 1 or < -1
            omega = (np.arcsin( n1n2 ) + np.arcsin( n2n3 ) + np.arcsin( n3n4 ) + np.arcsin( n4n1 ) ) * dprcvec/np.abs(dprcvec);
            
            wr+=omega/(4*np.pi)

    return 2*wr