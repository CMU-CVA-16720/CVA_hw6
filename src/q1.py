# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot

from math import floor, ceil
import os

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,) [x,y,z]

    rad : float
        The radius of the ball in cm

    light : numpy.ndarray
        The direction of incoming light [x,y,z]

    pxSize : float
        Pixel size in um

    res : numpy.ndarray
        The resolution of the camera frame [row,col]

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    ball_centr = np.array([res[0]//2, res[1]//2]) # row,col
    R = floor(rad*10**4/pxSize) # in pixels
    image = np.zeros(res)
    # Compute unit vector s
    s = np.copy(light)
    s = s/np.linalg.norm(s)
    for row in range(ball_centr[0]-R, ball_centr[0]+R):
        for col in range(ball_centr[1]-R, ball_centr[1]+R):
            # Shift coord
            y = ball_centr[0]-row
            x = col - ball_centr[1]
            # Check if inside circle
            if(x**2+y**2 <= R**2):
                ## Inside circle; render
                # Compute z
                z = (R**2-x**2-y**2)**(1/2)
                # Compute unit vector n
                n = np.array([x,y,z])
                n = n/np.linalg.norm(n)
                # Compute I
                I = np.dot(n,s)
                image[row,col] = I
                pass
            else:
                ## Outside circle; make black
                image[row,col] = 0
    image /= np.max(image)
    plt.imshow(image, cmap='gray')
    plt.show()

    image = None
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    I = None
    L = None
    s = None
    data_files = os.listdir(path)
    data_files.sort()
    I = np.array([])
    for file in data_files:
        file_path = os.path.join(path, file)
        if('input' in file):
            ## Image file
            # Import
            img = plt.imread(file_path)
            # make gray
            img = img[:,:,0]*0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11
            # Append
            if(I.size == 0):
                I = img.flatten()
                s = img.shape
            else:
                I = np.vstack((I, img.flatten()))
        else:
            L = np.transpose(np.load(file_path))

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None
    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None

    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """

    pass


if __name__ == '__main__':

    ## 1.b. Rendering
    # center = np.array([0,0,0])
    # rad = 0.75 # in cm
    # light_srcs = np.array([
    #     [1,1,1],
    #     [1,-1,1],
    #     [-1,-1,1]
    # ])
    # light_srcs = light_srcs/(3**(1/2))
    # pxSize = 7 # in um
    # res = np.array([3840,2160])
    # for i in range(0,light_srcs.shape[0]):
    #     light = light_srcs[i,:]
    #     renderNDotLSphere(center, rad, light, pxSize, res)
    ## 1.c. load data
    loadData()
    pass
