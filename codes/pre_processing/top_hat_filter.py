from skimage import io, morphology
from skimage.morphology import white_tophat, black_tophat, disk 
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as features
from scipy import ndimage
import cv2

filepath = 'data/'
your_file_path = filepath+'hsv_image.jpg'
tribolium = io.imread(your_file_path)
#tribolium = io.imread(your_file_path + 'MAX_Lund_18.0_22.0_Hours Z-projection t1.tif')

# setting the size of the minimum filter to be larger than the nuclei
size  = 25 


def diytophat(image, size=25):
    from scipy.ndimage import maximum_filter, minimum_filter
    minimum = minimum_filter(image,size)
    max_of_min = maximum_filter(minimum,size)
    tophat = image - max_of_min
    return tophat


def subtract_background(image, radius=50, light_bg=False):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # generate structuring element
    str_el = disk(radius)
     
    # use appropriate filter depending on the background colour
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)


def perform_CLAHE(image):
    #image = cv2.imread('hamster.jpg')
    #image = cv2.resize(image, (300, 400))
    
    # The initial processing of the image
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image_bw) + 30
    return final_img


def perform_Threshold(image):
    image = cv2.imread(filepath+'clahe_img.jpg')
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ordinary thresholding the same image
    ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)
    return ordinary_img


def perform_Tophat(image):  
    tribolium = image
    minimum_trib = ndimage.minimum_filter(tribolium, size)  
    orig_sub_min = tribolium - minimum_trib

    # getting the maximum of the minimum filtered image
    max_of_min_trib = ndimage.maximum_filter(minimum_trib, size)

    # subtraction from the original to obtain the top-hat filter
    tophat_trib = tribolium-max_of_min_trib

    #cv2.imwrite(filepath+'minimum_trib.jpg', minimum_trib)
    #cv2.imwrite(filepath+'orig_sub_min.jpg', orig_sub_min)
    return tophat_trib, max_of_min_trib


def perform_operations_and_save():
    tophat_max_min = diytophat(tribolium, 25)
    #tophat_black_bg = subtract_background(tribolium, 50, True)
    CLAHE_img = perform_CLAHE(tribolium)
    threshold_img = perform_Threshold(tribolium)
    tophat_trib, max_of_min_trib = perform_Tophat(tribolium)

    cv2.imwrite(filepath+'maximum_trib.jpg', max_of_min_trib)
    cv2.imwrite(filepath+'tophat_trib.jpg', tophat_trib)
    cv2.imwrite(filepath+'tophat_max_min.jpg', tophat_max_min)
    #cv2.imwrite(filepath+'tophat_black_bg.jpg', tophat_trib)
    cv2.imwrite(filepath+'CLAHE_img.jpg', CLAHE_img)
    #cv2.imwrite(filepath+'threshold_img.png', threshold_img)




