# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
import numpy as np
from matplotlib import pyplot as plt
from utils import integrateFrankot
import skimage.color
import cv2

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
        The resolution of the camera frame [width,height]

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """
    ball_centr = np.array([res[1]//2, res[0]//2, 0]) # row, col, center of frame
    ball_centr += np.array([center[1], center[0], center[2]]) # coordinate offset
    R = floor(rad*10**4/pxSize) # in pixels
    image = np.zeros((res[1],res[0]))
    # Compute unit vector s
    s = np.copy(light)
    s = s/np.linalg.norm(s)
    row_min = max(ball_centr[0]-R,0)
    row_max = min(ball_centr[0]+R, res[1]-1)
    col_min = max(ball_centr[1]-R,0)
    col_max = min(ball_centr[1]+R, res[0]-1)
    for row in range(row_min, row_max):
        for col in range(col_min, col_max):
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
    image[image<0] = 0 # set negative intensity to 0
    image /= np.max(image)
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
            img = cv2.imread(file_path, -1)
            # get intensities
            img = skimage.color.rgb2xyz(img)[:,:,1]
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
    # Use left pseudo-inverse
    B = np.linalg.inv(L@np.transpose(L))@L@I
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

    albedos = np.linalg.norm(B,axis=0)
    normals = B/albedos
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
    albedoIm = albedos.reshape(s)
    normalIm = np.stack((
        normals[0,:].reshape(s),
        normals[1,:].reshape(s),
        normals[2,:].reshape(s)), axis=2)
    plt.imshow(albedoIm,cmap='coolwarm')
    plt.show()
    plt.imshow(normalIm,cmap='rainbow')
    plt.show()
    

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
    n1 = normals[0,:].reshape(s)
    n2 = normals[1,:].reshape(s)
    n3 = normals[2,:].reshape(s)
    dfdx = n1/n3
    dfdy = n2/n3
    surface = integrateFrankot(dfdx, dfdy)
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
    # Create grid
    x = np.arange(0, surface.shape[1])
    y = np.arange(0, surface.shape[0])
    X, Y = np.meshgrid(x,y)
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, surface, cmap='coolwarm')
    plt.show()
    


    pass


if __name__ == '__main__':

    ## 1.b. Rendering
    if False:
        center = np.array([0,0,0]) # in pixels
        rad = 0.75 # in cm
        light_srcs = np.array([
            [1,1,1],
            [1,-1,1],
            [-1,-1,1]
        ])
        light_srcs = light_srcs/(3**(1/2))
        pxSize = 7 # in um
        res = np.array([3840,2160])
        for i in range(0,light_srcs.shape[0]):
            light = light_srcs[i,:]
            rendr = renderNDotLSphere(center, rad, light, pxSize, res)
            plt.imshow(rendr, cmap='gray')
            plt.show()

    ## 1.c. load data
    I, L, s = loadData()
    ## 1.e. Pseudonormal & Albedo
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    # displayAlbedosNormals(albedos, normals, s)
    ## 1.i. Depth
    surface = estimateShape(normals, s)
    plotSurface(surface)
    
    pass
