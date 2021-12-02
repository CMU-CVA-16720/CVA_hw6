# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    B = None
    L = None
    # SVD to get U, S, VT
    U, S, VT = np.linalg.svd(I, False)
    # Force rank constraint by modifying S
    S[3:] = 0
    # Recompute I
    I2 = U@np.diag(S)@VT
    # Recompute U, VT
    U2, _, VT2 = np.linalg.svd(I2, False)
    # Breakout into B and L
    L = U[0:3, :]
    B = VT[0:3, :]

    return B, L


if __name__ == "__main__":

    # Get data
    I, L, s = loadData()
    # Compute ambiguous data
    Bhat, Lhat = estimatePseudonormalsUncalibrated(I)
    # Compare
    print('L = \n{}'.format(L))
    print('Lhat = \n{}'.format(Lhat))
    # 2.d
    albedos, normals = estimateAlbedosNormals(Bhat)
    surface = estimateShape(normals, s)
    if False:
        plotSurface(surface)
    # 2.e.
    normals2 = enforceIntegrability(normals, s)
    surface2 = estimateShape(normals2, s)
    if False:
        plotSurface(surface2)
    # 2.f.
    # Parameters
    mu = 0
    v = 0
    lambda_ = 1 # > 0
    # G
    G = np.eye(3)
    G[2,:] = np.array([mu, v, lambda_])
    # New B
    albedos, normals = estimateAlbedosNormals(Bhat)
    normals = enforceIntegrability(normals, s)
    normals_new = np.linalg.inv(np.transpose(G))@normals
    normals_new = normals_new/np.linalg.norm(normals_new,axis=0)
    surface = estimateShape(normals_new, s)
    plotSurface(surface)
    pass
