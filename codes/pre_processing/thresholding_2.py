import glob

import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
import cv2

filepath = 'data/'


soybean = iio.imread(uri=filepath+"soybean_test.jpg")

def perform_thresholding(image):
    image_to_use = image
    '''
    fig, ax = plt.subplots()
    ax.imshow(image_to_use)
    '''

    # convert the image to grayscale
    gray_image = ski.color.rgb2gray(image_to_use)

    # blur the image to denoise
    blurred_image = ski.filters.gaussian(gray_image, sigma=1.0)

    # show the histogram of the blurred image
    '''
    histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
    fig, ax = plt.subplots()
    ax.plot(bin_edges[0:-1], histogram)
    ax.set_title("Graylevel histogram")
    ax.set_xlabel("gray value")
    ax.set_ylabel("pixel count")
    ax.set_xlim(0, 1.0)
    '''

    # perform automatic thresholding
    t = ski.filters.threshold_otsu(blurred_image)
    print("Found automatic threshold t = {}.".format(t))

    #Found automatic threshold t = 0.6392454549881862.

    # create a binary mask with the threshold found by Otsu's method
    binary_mask = blurred_image > t

    '''
    fig, ax = plt.subplots()
    ax.imshow(binary_mask, cmap="gray")
    '''

    # apply the binary mask to select the foreground
    selection = image_to_use.copy()
    selection[~binary_mask] = 0

    '''
    fig, ax = plt.subplots()
    ax.imshow(selection)
    '''
    return selection