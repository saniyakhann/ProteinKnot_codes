import numpy as np
from numba import prange, njit

@njit
def compute_writhe_matrix(ring1, ring2):
 
    n1, n2 = ring1.shape[0], ring2.shape[0]
    matrix = np.zeros((n1, n2))
    
    # Compute only upper triangle for efficiency
    for i in prange(n1):
        for j in prange(i, n2):
            value = compute_single_writhe(ring1, ring2, i, j)
            matrix[i,j] = value
            if i < j:
                matrix[j,i] = value  # Keep symmetry
                
    return matrix

@njit
def compute_single_writhe(ring1, ring2, i, j):
   
    wr = 0
    epsilon = 1e-12
    
    #one segment pair (no loops)
    # Segment 1: from point i-1 to i
    # Segment 2: from point j-1 to j
    
    one = ring1[np.mod(i-1, ring1.shape[0]), :]
    two = ring1[np.mod(i, ring1.shape[0]), :]
    three = ring2[np.mod(j-1, ring2.shape[0]), :]
    four = ring2[np.mod(j, ring2.shape[0]), :]
    
  
    r12 = two - one
    r34 = four - three
    r23 = three - two
    r13 = three - one
    r14 = four - one
    r24 = four - two
    
    n1 = np.cross(r13, r14)
    n1_norm = np.linalg.norm(n1)
    if n1_norm < epsilon:
        return 0
    n1 = n1 / n1_norm
    
    n2 = np.cross(r14, r24)
    n2_norm = np.linalg.norm(n2)
    if n2_norm < epsilon:
        return 0
    n2 = n2 / n2_norm
    
    n3 = np.cross(r24, r23)
    n3_norm = np.linalg.norm(n3)
    if n3_norm < epsilon:
        return 0
    n3 = n3 / n3_norm
    
    n4 = np.cross(r23, r13)
    n4_norm = np.linalg.norm(n4)
    if n4_norm < epsilon:
        return 0
    n4 = n4 / n4_norm
    
    n1n2 = np.dot(n1, n2)
    n2n3 = np.dot(n2, n3)
    n3n4 = np.dot(n3, n4)
    n4n1 = np.dot(n4, n1)
    

    n1n2 = max(min(n1n2, 1.0 - epsilon), -1.0 + epsilon)
    n2n3 = max(min(n2n3, 1.0 - epsilon), -1.0 + epsilon)
    n3n4 = max(min(n3n4, 1.0 - epsilon), -1.0 + epsilon)
    n4n1 = max(min(n4n1, 1.0 - epsilon), -1.0 + epsilon)
    
    cvec = np.cross(r34, r12)
    dprcvec = np.dot(cvec, r13)
    
    if abs(dprcvec) < epsilon:
        return 0
    
    omega = (np.arcsin(n1n2) + np.arcsin(n2n3) + np.arcsin(n3n4) + np.arcsin(n4n1)) * dprcvec/np.abs(dprcvec)
    
    return omega / (4 * np.pi)  #no 2 factor here 
