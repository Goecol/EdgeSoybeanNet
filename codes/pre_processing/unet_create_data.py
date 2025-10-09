import cv2
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torchvision.transforms.functional as Ff
import sys
import os
import time 

sys.path.append(os.path.abspath("codes"))
from UNetLite import UNetLite  

batch_size = 1
height = 300
width = 300


class SoybeanPodDataset(Dataset):
    def __init__(self, image_paths,  transform=None):
        """
        Args:
            image_paths (list of str): List of file paths to input images
            mask_paths (list of str): List of file paths to corresponding masks
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        self.image_paths = sorted(self.image_paths)  # Sort image file paths
        #self.mask_paths = sorted(self.mask_paths)    # Sort label mask file paths
        image_path = self.image_paths[idx]

        # Load the image and mask
        #image = Image.open(self.image_paths[idx]).convert("RGB")
        image = Image.open(self.image_paths[idx])

        mask_trans = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        #transforms.Normalize([0.5], [0.5]) # imagenet   #one channel
          ])


        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)

        return image, image_path
    

def getDevice():
    device = None
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_on_gpu = torch.cuda.is_available()
    #train_on_gpu = torch.backends.mps.is_available()
    #train_on_gpu = False

    if not train_on_gpu:
        print('CUDA/MPS is not available.  Training on CPU ...')
        device = torch.device("cpu")
    else:
        print('CUDA/MPS is available!  Training on GPU ...')
        #device = torch.device("mps")
        device = torch.device("cuda")
    
    return device

def get_image_filepaths(directory):
    """Returns a list of filepaths for all images in the given directory."""

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_paths = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and any(filename.endswith(ext) for ext in image_extensions):
            image_paths.append(filepath)

    return image_paths

def get_data_loaders(image_dir):
    trans = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # imagenet
    ])
    # # Create another simulation dataset for test
    test_image_paths = get_image_filepaths(image_dir)

    test_dataset = SoybeanPodDataset(test_image_paths, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return test_loader

def show_output_plot(predicted_mask, inputs, count, i):
            # Visualize the result
            plt.imshow(predicted_mask[0, 0].cpu().numpy(), cmap='gray')
            plt.savefig("plt_predicted_mask_" + str(count) + ".png", dpi=300, bbox_inches='tight')
            plt.show()

            numpy_input = inputs[i].permute(1,2,0).cpu().numpy()
            numpy_predicted = predicted_mask[0,0].cpu().numpy()

            # Plot the images in a single row with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Show the test image     
            axes[0].imshow(numpy_input)
            axes[0].set_title("Test Image")
            #axes[0].axis('off')  # Hide axis for better visualization

            # Show the predicted mask
            axes[2].imshow(numpy_predicted, cmap='gray')
            axes[2].set_title("Predicted Mask")
            #axes[2].axis('off')  # Hide axis

            # Display the plot
            plt.tight_layout()
            plt.savefig(f"plt_predicted_mask_{count}_comp.png", dpi=300, bbox_inches='tight')
            plt.show()


def predict_model(image_dir, new_image_dir):
    # The new image dir is the directory for saving the new predicted image

    in_channels = 3
    num_class = 2
    num_out_channels = 1 # for binary segmentation, the output channel = 1
    response = False
    image_size = (width, height)
 
    # Get the current device
    device = getDevice()

    # load the structure of the model first before loading trained weights from saved trained model
    model = UNetLite(num_class, num_out_channels, image_size).to(device)

    # Load the saved model weights
    model.load_state_dict(torch.load('my_trained_model.pth'))

    # Perform model testing and prediction
    model.eval()  # Set model to the evaluation mode

    # Load the test data using the image directory
    test_loader = get_data_loaders(image_dir)

    i = 0
    with torch.no_grad():
        for count, (inputs, image_paths) in enumerate(test_loader, start=1):   # Replace test_loader with your test data
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Apply sigmoid to get probabilities and threshold to get binary mask
            predicted_mask = torch.sigmoid(outputs) 
            predicted_mask_numpy = predicted_mask[0, 0].cpu().numpy()
            predicted_mask_numpy = (predicted_mask_numpy * 255).astype(np.uint8)
            # Save predicted mask as an image
            output_mask_image = Image.fromarray(predicted_mask_numpy)  # Convert to uint8
            image_name_without_ext = os.path.splitext(os.path.basename(image_paths[0]))[0]
            mask_filename = image_name_without_ext + ".jpg"
            save_path = os.path.join(new_image_dir, mask_filename)
            print(mask_filename)
            print(save_path)
            output_mask_image.save(save_path)
            response = True
            #show_output_plot(predicted_mask, inputs, count, i)

    return response
