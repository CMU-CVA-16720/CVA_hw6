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
    U2, S2, VT2 = np.linalg.svd(I2, False)
    # Breakout into B and L
    L = (U2@np.diag(S2))[0:3, :]
    B = VT2[0:3, :]

    return B, L


if __name__ == "__main__":

    # Get data
    I, L, s = loadData()
    # Compute ambiguous data
    Bhat, Lhat = estimatePseudonormalsUncalibrated(I)
    # 2.b.
    albedos, normals = estimateAlbedosNormals(Bhat)
    if False:
        displayAlbedosNormals(albedos, normals, s)
    # 2.c.
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
    mu, v, lambda_ = 0,0,1
    mu, v, lambda_ = 20,0,1
    mu, v, lambda_ = -10,0,1
    mu, v, lambda_ = 0,20,1
    mu, v, lambda_ = 0,-10,1
    mu, v, lambda_ = 0,0,20
    mu, v, lambda_ = 0,0,0.0000001
    # G
    G = np.eye(3)
    G[2,:] = np.array([mu, v, lambda_])
    # New B
    albedos, normals = estimateAlbedosNormals(Bhat)
    normals = enforceIntegrability(normals, s)
    normals_new = np.linalg.inv(np.transpose(G))@normals
    normals_new = normals_new/np.linalg.norm(normals_new,axis=0)
    surface = estimateShape(normals_new, s)
    if True:
        plotSurface(surface)
    pass
