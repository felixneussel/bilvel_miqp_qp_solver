import numpy as np



def concatenateDiagonally(H,G):
    """
    Concatenates to matrices diagonally with zeros on the off-diagonals
    """
    H_rows, H_cols = H.shape
    G_rows, G_cols = G.shape

    matrix = np.zeros((H_rows+G_rows,H_cols+G_cols))
    matrix[:H_rows,:H_cols] = H
    matrix[H_rows:,H_cols:] = G
    return matrix

def concatenateHorizontally(A,B):
    A_rows,A_cols = A.shape
    B_rows,B_cols = B.shape
    if A_rows != B_rows:
        raise ValueError('Input matrices need to have the same number of rows')
    matrix = np.zeros((A_rows,A_cols+B_cols))
    matrix[:,:A_cols] = A
    matrix[:,A_cols:] = B
    return matrix

def getLowerBound():
    """
    Returns a lower bound for the w_jr constraints.
    """
    #Kleinert et al used 1e5 for all big-Ms
    return -1e5

def getUpperBound():
    """
    Returns an upper bound for the w_jr constraints.
    """
    #Kleinert et al used 1e5 for all big-Ms
    return 1e5