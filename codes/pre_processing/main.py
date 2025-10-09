from skimage import io, morphology
from skimage.morphology import white_tophat, black_tophat, disk 
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as features
from scipy import ndimage
import cv2
import glob
import os
from os import listdir
from PIL import Image

import top_hat_filter
import thresholding_2
import image_create_new_main as image_create_cont
import image_create_contours_using_dots_km_processing as image_create_con_using_dots
import image_extract_points_json as extract 
import image_create_new_main_mask as image_create_cont_mask
import unet_create_data as unet_create


filepath = '/home/johnbosco/soybean/dataset/'
new_filepath = '/home/johnbosco/soybean/dataset/pre_processed/'
# for the new file structure
new_filepath_1 = '/home/johnbosco/soybean/dataset/original/pre_processed/'
'''
filepath = '/home/johnbosco/image_processing/code/data/testing'
new_filepath = '/home/johnbosco/image_processing/code/data/'
'''

prefix = 'pre_'
test_dir = 'test/'
train_dir = 'train/'
val_dir =  'val/'
default_dir = 'pre_processed/'


inner_file_dir = '001.lessthan40/'
inner_file_dirs = ['001.lessthan40/', '002.41to80/', '003.81to120/', '004.121to160/', '005.161to200/', '006.201to240/', '007.241to280/', '008.morethan281/']
#inner_file_dirs = ['004.121to160/', '005.161to200/', '006.201to240/', '007.241to280/', '008.morethan281/']
#inner_file_dirs = ['007.241to280/', '008.morethan281/']
'''
inner_file_dir = 'images/'
inner_file_dirs = ['images/']
'''


def getImageName(file_location):
    filename = file_location.split('/')[-1]
    location = file_location.split('/')[0:-1]
    return filename


def getImages_Process(imgdir, pre_process_type, model_type):
    ext = ['png', 'jpg', 'gif', 'bmp']    # These are acceptable image formats

    print(imgdir)
    print(model_type)

    model_filepath = train_dir
    if model_type == 'train':
        model_filepath = train_dir
    elif model_type == 'val':
        model_filepath = val_dir
    elif model_type == 'test':
        model_filepath = test_dir
    else:
        model_filepath = default_dir
  
    for inner_file_dir in inner_file_dirs:
        full_filepath_name = imgdir + model_filepath + inner_file_dir 
        for filename in os.listdir(full_filepath_name):
            print(filename)  

            img = cv2.imread(full_filepath_name+filename) 
            new_image = None
            if(pre_process_type == "type7"):
               sigma=1.0
               strength=2.0
               new_filepath = '/home/johnbosco/soybean/dataset/pre_processed_sharpened_image1/'
               # Convert to float to prevent overflow issues
               image = img.astype(np.float32)

               # Apply Gaussian Blur
               blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=sigma, sigmaY=sigma)

               # Create the sharpened image
               sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

               # Clip values to valid range and convert back to uint8
               sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
               new_image = sharpened
            elif(pre_process_type == "type5"):
               new_filepath = '/home/johnbosco/soybean/dataset/pre_processed_cont_mask/'
               contour_box_image = image_create_cont_mask.perform_contour_box_process_mask(img)        
               new_image = contour_box_image[0]
            elif(pre_process_type == "type4"):
               contour_box_image = image_create_cont.perform_contour_box_process(img)        
               new_image = contour_box_image
            elif(pre_process_type == "type3"):
               thresholding_image = thresholding_2.perform_thresholding(img)
               #top_hat_image = top_hat_filter.perform_Tophat(thresholding_image)
               #new_image = top_hat_image
               new_image = thresholding_image
            elif(pre_process_type == "type1"):
               top_hat_image = top_hat_filter.perform_Tophat(img)
               thresholding_image = thresholding_2.perform_thresholding(top_hat_image)
               new_image = thresholding_image
            else:
               thresholding_image = thresholding_2.perform_thresholding(img)
               top_hat_image = top_hat_filter.perform_Tophat(thresholding_image)
               new_image = top_hat_image
            new_filename = new_filepath+ model_filepath + inner_file_dir + prefix + pre_process_type + "_" + filename
            print(new_filename)
            cv2.imwrite(new_filename, new_image)            


#This processes different file structure from the previous datasets
def getImages_Process_Contour_Dots(imgdir, pre_process_type, model_type):
    print("In here")
    ext = ['png', 'jpg', 'gif', 'bmp']    # These are acceptable image formats

    print(imgdir)
    print(model_type)
    model_filepath = train_dir
    if model_type == 'train':
        model_filepath = train_dir
    elif model_type == 'val':
        model_filepath = val_dir
    elif model_type == 'test':
        model_filepath = test_dir
    else:
        model_filepath = default_dir
  
        
    full_filepath_name = imgdir + model_filepath 
    print(full_filepath_name)
    for filename in os.listdir(full_filepath_name):
        print(filename)  
  
        img = cv2.imread(full_filepath_name+filename) 
        filename_without_ext = os.path.splitext(filename)[0]
        full_filepath_name_json = imgdir + "json/"
        points = extract.extract_points_from_json(full_filepath_name_json+filename_without_ext+".json")
        new_image = None
        if(pre_process_type == "type6"):
           contour_box_image = image_create_con_using_dots.perform_contour_box_process_using_dots(img, points)        
           new_image = contour_box_image[1]
        else:
            thresholding_image = thresholding_2.perform_thresholding(img)
            top_hat_image = top_hat_filter.perform_Tophat(thresholding_image)
            new_image = top_hat_image
        #new_filename = new_filepath_1+ model_filepath + prefix + pre_process_type + "_" + filename
        new_filename = new_filepath_1+ model_filepath + filename
        print(new_filename)
        cv2.imwrite(new_filename, new_image)            


def convert_mask_to_rgb(imgdir, output_dir, model_type):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ext = ['png', 'jpg', 'gif', 'bmp', 'jpeg']    # These are acceptable image formats

    print(imgdir)
    print(model_type)

    model_filepath = train_dir
    if model_type == 'train':
        model_filepath = train_dir
    elif model_type == 'val':
        model_filepath = val_dir
    elif model_type == 'test':
        model_filepath = test_dir
    else:
        model_filepath = default_dir

    for inner_file_dir in inner_file_dirs:
        full_filepath_name = imgdir + model_filepath + inner_file_dir 
        # Loop through all images in the input directory
        for filename in os.listdir(full_filepath_name):
            input_path = os.path.join(full_filepath_name, filename)

            # Only process image files (ensure it's an image file)
            if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    # Open the image file
                    with Image.open(input_path) as img:
                        # Convert Mask (e.g., grayscale or single-channel) to RGB
                        img_rgb = img.convert("RGB")

                        # Save the converted image in the output directory
                        full_filepath_name_output = output_dir + model_filepath + inner_file_dir 
                        full_filepath_name_output = os.path.join(full_filepath_name_output, filename)
                        img_rgb.save(full_filepath_name_output)

                        print(f"Converted {filename} to RGB and saved to {full_filepath_name_output}")
                except Exception as e:
                    print(f"Could not process {filename}: {e}")        


#This processes different file structure from the previous datasets
def getImages_Predict_UNet(imgdir, pre_process_type, model_type):
    new_filepath = filepath+"pre_processed_unet/"
    print("In here")
    ext = ['png', 'jpg', 'gif', 'bmp']    # These are acceptable image formats

    print(imgdir)
    print(model_type)
    model_filepath = train_dir
    if model_type == 'train':
        model_filepath = train_dir
    elif model_type == 'val':
        model_filepath = val_dir
    elif model_type == 'test':
        model_filepath = test_dir
    else:
        model_filepath = default_dir
  
    for inner_file_dir in inner_file_dirs:
        full_filepath_name = imgdir + model_filepath + inner_file_dir
        new_full_filepath_name = new_filepath+ model_filepath + inner_file_dir     
        response = unet_create.predict_model(full_filepath_name, new_full_filepath_name)
        


#Processing original large datasets of 112 images
getImages_Process_Contour_Dots('/home/johnbosco/soybean/dataset/original/', 'type6', 'train')
getImages_Process_Contour_Dots('/home/johnbosco/soybean/dataset/original/', 'type6', 'val')
getImages_Process_Contour_Dots('/home/johnbosco/soybean/dataset/original/', 'type6', 'test')

'''
# Processing initial datasets using contour pre-processing to create masks
getImages_Predict_UNet('/home/johnbosco/soybean/dataset/', 'type8', 'train')
#getImages_Predict_UNet('/home/johnbosco/soybean/dataset/', 'type8', 'test')
#getImages_Predict_UNet('/home/johnbosco/soybean/dataset/', 'type8', 'val')
'''