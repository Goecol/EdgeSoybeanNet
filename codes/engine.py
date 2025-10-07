import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torchvision.transforms.functional as Ff
import os
import torch.nn as nn
from torchinfo import summary
import shutil
from scipy.ndimage import label, center_of_mass
from scipy import ndimage as ndi
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import json

# Import the Json file for processing soybean cordinates
import image_extract_points_json as json_point_extract


torch.cuda.empty_cache()

train_dir = "/home/johnbosco/soybean/dataset/original/train/"
train_mask_dir = "/home/johnbosco/soybean/dataset/original/pre_processed/train/"
train_aug_dir = "/home/johnbosco/soybean/dataset/original/augmented_mul/train/"
train_mask_aug_dir = "/home/johnbosco/soybean/dataset/original/pre_processed/augmented_mul/train/"
val_aug_dir = "/home/johnbosco/soybean/dataset/original/augmented_mul/val/"
val_mask_aug_dir = "/home/johnbosco/soybean/dataset/original/pre_processed/augmented_mul/val/"
val_dir = "/home/johnbosco/soybean/dataset/original/val/"
val_mask_dir = "/home/johnbosco/soybean/dataset/original/pre_processed/val/"
test_dir = "/home/johnbosco/soybean/dataset/original/test/"
test_mask_dir = "/home/johnbosco/soybean/dataset/original/pre_processed/test/"
pod_json_get_path = "/home/johnbosco/soybean/dataset/original/json/"
#pod_json_save_path = "/home/johnbosco/soybean/dataset/original/json/json_threshold_data/"
pod_json_save_path = os.path.dirname(os.path.abspath(__file__))
train_json_threshold_filename = "train_json_threshold.json"
val_json_threshold_filename = "val_json_threshold.json"
script_dir = os.path.dirname(os.path.abspath(__file__))


batch_size = 4
threshold_batch_size = 4
height = 560  # set the height of the input size of the image to the model
width = 560  # set the width of the input size of the image to the model
train_loss = []
val_loss = []
train_acc_list = []
val_acc_list = []
train_dice_loss_list = []
val_dice_loss_list = []
train_dice_score_list, val_dice_score_list = [], []
train_iou_list, train_precision_list = [], []
val_iou_list, val_precision_list = [], []
train_threshold_loss_list, val_threshold_loss_list = [], []
thresholds_tensor_old = torch.tensor([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99])
thresholds_tensor = torch.tensor([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
num_thresholds = 10
# Seed=52 and Seed=50 are best for now giving excellent 45% accuracy.
#seed = 50
seed = 21
seed1 = 24


# Set global font size for all text elements
plt.rcParams.update({
    'font.size': 16,        # default text size
    'axes.titlesize': 21,   # subplot title size
    'axes.labelsize': 19,   # x/y label size
    'xtick.labelsize': 17,  # x tick label size
    'ytick.labelsize': 17,  # y tick label size
    'legend.fontsize': 17   # legend font size
})


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_seed_1(seed1=25):
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)
    torch.cuda.manual_seed_all(seed1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed1)    


def make_loader(dataset, batch_size, shuffle, seed, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=lambda _: np.random.seed(seed),
        generator=torch.Generator().manual_seed(seed)
    )


# Optional: Your own method to compute best threshold
def compute_best_threshold(pred_mask, gt_mask, thresholds=thresholds_tensor):
    # Ensure pred_mask is a NumPy array
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    thresholds = thresholds.numpy()

    best_thresh = 0.5
    best_score = -1
    for t in thresholds:
        bin_mask = (pred_mask > t).astype(np.uint8)
        intersection = np.logical_and(bin_mask, gt_mask).sum()
        union = np.logical_or(bin_mask, gt_mask).sum()
        iou = intersection / (union + 1e-8)
        if iou > best_score:
            best_score = iou
            best_thresh = t
    return best_thresh


def get_best_threshold_usingPodCount(pred_mask, pod_count, thresholds):
    thresholds = thresholds.numpy()

    threshold_list = []
    pred_pod_count_list = []
    best_thresh = 0.5
    best_pred_pod_count = 2000
    # Use a large number for the smallest difference to start checking
    smallest_difference = 500000
    for t in thresholds:
        binary_mask = (pred_mask > t).astype(np.uint8)
        # This computes the num_pods or connected component in the binary mask
        pred_labeled_mask, pred_mask_pod_count = label(binary_mask)
        difference = abs(pred_mask_pod_count - pod_count)
        pred_pod_count_list.append(pred_mask_pod_count)
        threshold_list.append(t)

        if  difference < smallest_difference:
            smallest_difference = difference
            best_thresh = t
            best_pred_pod_count = pred_mask_pod_count

    return best_thresh, best_pred_pod_count, threshold_list, pred_pod_count_list

def compute_best_threshold_usingPodCount(pred_mask, pod_count, target_mask=None, thresholds=thresholds_tensor):
    # Ensure pred_mask is a NumPy array
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    threshold_data = []
    best_thresh, best_pred_pod_count, threshold_list, pred_pod_count_list = get_best_threshold_usingPodCount(pred_mask, pod_count, thresholds)

    # Step 2: Convert to binary mask (everything > 0 is considered foreground)
    if target_mask is not None:
        if isinstance(target_mask, torch.Tensor):
            target_mask_np = target_mask.cpu().numpy()
        else:
            target_mask_np = np.array(target_mask)

        target_mask_pod_count = -1
        target_mask_best_thresh = -0.5
        if target_mask_np.max() < 1.0:
            #target_binary_mask = (target_mask_np > 0.5).astype(np.uint8)
            target_mask_best_thresh, target_mask_pod_count, _, _ = get_best_threshold_usingPodCount(target_mask_np, pod_count, thresholds)
        else:
           target_binary_mask = (target_mask_np > 127).astype(np.uint8)
           target_labeled_mask, target_mask_pod_count = label(target_binary_mask)

    else:
       target_mask_pod_count = -1

    threshold_data.append(best_thresh)
    threshold_data.append(best_pred_pod_count)
    threshold_data.append(target_mask_best_thresh)
    threshold_data.append(target_mask_pod_count)
    threshold_data.append(threshold_list)
    threshold_data.append(pred_pod_count_list)

    return threshold_data


def threshold_to_class_index(threshold_value, allowed_thresholds):
    """Convert float threshold to closest class index."""
    idx = (np.abs(allowed_thresholds - threshold_value)).argmin()
    return int(idx)

class ThresholdDataSetWithMask():
    def __init__(self, image_paths, mask_paths, unet_model, transform, json_get_pathname, device='cuda', shuffle=False, image_size=(height, width)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.unet_model = unet_model
        self.transform = transform
        self.json_get_pathname = json_get_pathname
        self.image_size = image_size
        self.device = device
        self.shuffle = shuffle

        self.data = []

        dataset = SoybeanPodDataset(self.image_paths, self.mask_paths, transform=self.transform)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)
        dataloader   = make_loader(dataset, batch_size=batch_size, shuffle=self.shuffle, seed=seed1)

        with torch.no_grad():
            for count, (inputs, target_mask, data_image_paths) in enumerate(dataloader, start=1):
                inputs = inputs.to(self.device)
                mask_logits = self.unet_model(inputs)
                pred_mask = torch.sigmoid(mask_logits)

                for i in range(inputs.size(0)):
                    image_path = data_image_paths[i]
                    image_name = os.path.splitext(os.path.basename(image_path))[0]

                    with open(self.json_get_pathname, 'r') as f:
                        json_file_data = json.load(f)

                    pred_mask_threshold = 0.5
                    if json_file_data is not None and os.path.getsize(json_get_pathname) > 0:
                        pred_mask_threshold = json_file_data[image_name]["pred_mask_threshold"]

                    print("image_name: ", image_name)
                    print("pred_mask: ", pred_mask[i])
                    print("pred_mask_threshold: ", pred_mask_threshold)
                    self.data.append((pred_mask[i], pred_mask_threshold))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
       

class ThresholdDataSetWithMaskClassifier_Features():
    def __init__(self, image_paths, mask_paths, unet_model, transform, json_get_pathname, device='cuda', data_input_type="features", shuffle=False, image_size=(572, 572)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.unet_model = unet_model
        self.transform = transform
        self.json_get_pathname = json_get_pathname
        self.image_size = image_size
        self.device = device
        self.shuffle = shuffle

        self.data = []

        dataset = SoybeanPodDataset(self.image_paths, self.mask_paths, transform=self.transform)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)
        dataloader   = make_loader(dataset, batch_size=batch_size, shuffle=self.shuffle, seed=seed1)

        # Load the JSON once outside the loop
        with open(self.json_get_pathname, 'r') as f:
            json_file_data = json.load(f)

        with torch.no_grad():
            for count, (inputs, target_mask, data_image_paths) in enumerate(dataloader, start=1):
                inputs = inputs.to(self.device)
                mask_logits = self.unet_model(inputs)
                pred_mask = torch.sigmoid(mask_logits)

                for i in range(inputs.size(0)):
                    image_path = data_image_paths[i]
                    image_name = os.path.splitext(os.path.basename(image_path))[0]

                    # Default threshold value
                    pred_mask_threshold = 0.5

                    if image_name in json_file_data:
                        pred_mask_threshold = json_file_data[image_name]["pred_mask_threshold"]

                    class_index = threshold_to_class_index(pred_mask_threshold, thresholds_tensor)
                    print("class_index: ", class_index)      
                    print("pred_mask: ", pred_mask[i])
                    self.data.append((pred_mask[i], class_index))  # Store mask and class index

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]            


class BestThresholdPredMasks():
    def __init__(self, image_dir, mask_dir, unet_model, transform, json_get_path, json_save_path,  device='cuda', image_size=(height, width)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.unet_model = unet_model.to(device).eval()
        self.image_size = image_size
        self.transform = transform
        self.device = device

        self.data = []
        #Defining the empty data to dump json
        self.json_file = ""
        print("Computing the best threshold with pod count")
        
        self.image_paths = get_image_filepaths(self.image_dir)
        self.mask_paths = get_image_filepaths(self.mask_dir)
        dataset = SoybeanPodDataset(self.image_paths, self.mask_paths, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        i = 0
        with torch.no_grad():
            for count, (inputs, target_mask, data_image_paths) in enumerate(dataloader, start=1):   # Replace test_loader with your test data
              inputs = inputs.to(self.device)
              mask_logits = self.unet_model(inputs)
              pred_mask = torch.sigmoid(mask_logits)

              for i in range(inputs.size(0)):
                 #mask_prob = mask_probs[i]
                 image_path = data_image_paths[i]
                 image_name = os.path.splitext(os.path.basename(image_path))[0]
                 file_name = image_name
                 if "_aug" in image_name:
                    file_name =  image_name.split("_aug")[0]
                 # The image and json files have the same base name for 
                 json_filename = os.path.join(json_get_path, file_name + ".json")
                 
                 if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
                        with open(json_filename, 'r') as f:
                            json_file_data = json.load(f)

                        # Checking if the json file is loaded and it is not None
                        if (json_file_data is not None) and (file_name in json_file_data):
                            pod_count = json_file_data[file_name]["pod_count"]
                        else:
                            pod_coordinates = json_point_extract.extract_points_from_json(json_filename)
                            pod_count = len(pod_coordinates)
                 else:
                        pod_coordinates = json_point_extract.extract_points_from_json(json_filename)
                        pod_count = len(pod_coordinates)

                 threshold_data = compute_best_threshold_usingPodCount(pred_mask[i], pod_count, target_mask[i])

                 image_id = image_name
                 entry_data = {
                     "pod_count" : int(pod_count),
                     "pred_mask_threshold" : float(threshold_data[0]),
                     "pred_mask_pod_count" : int(threshold_data[1]),
                     "target_mask_threshold" : float(threshold_data[2]),
                     "target_mask_pod_count" : int(threshold_data[3]),
                     "pred_threshold_list" : [float(x) for x in threshold_data[4]],
                     "pred_pod_count_list" : [int(x) for x in threshold_data[5]]
                 }

                 print("Image All data - ", image_id, entry_data)
                 try:
                   # Read the json file in order to load the data
                   with open(json_save_path, 'r') as f:
                     all_data = json.load(f)
                 except FileNotFoundError:
                     all_data = {}

                 all_data[image_id] = entry_data

                 # Write the updated data back to the json file
                 with open(json_save_path, 'w') as f:
                     json.dump(all_data, f, indent=4)

                 self.data.append({
                      "image_id" : image_name,
                      **entry_data
                 })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_by_image_id(self, image_id):
        for item in self.data:
            if image_id == item[image_id]:
                return item
        return None

                    

def createThresholdDataLoaderWithMask(unet_model, train_json_get_pathname, val_json_get_pathname, transform, device, cnn_type="classifier", data_augment=True):
    # File cache paths
    #train_cache = os.path.join(pod_json_save_path, f"cached_train_{cnn_type}.pt")
    #val_cache = os.path.join(pod_json_save_path, f"cached_val_{cnn_type}.pt")
    train_cache = os.path.join(script_dir, f"cached_train_{cnn_type}.pt")
    val_cache = os.path.join(script_dir, f"cached_val_{cnn_type}.pt")

    # Get image/mask paths
    image_paths = get_image_filepaths(train_dir)
    mask_paths = get_image_filepaths(train_mask_dir)
    val_image_paths = get_image_filepaths(val_dir)
    val_mask_paths = get_image_filepaths(val_mask_dir)

    if data_augment:
        image_paths = get_image_filepaths(train_aug_dir)
        mask_paths = get_image_filepaths(train_mask_aug_dir)
        val_image_paths = get_image_filepaths(val_aug_dir)
        val_mask_paths = get_image_filepaths(val_mask_aug_dir)

    # Get the directory where the current script is located
    
    # Full JSON paths
    train_json_get_pathname = os.path.join(pod_json_save_path, train_json_get_pathname)
    val_json_get_pathname = os.path.join(pod_json_save_path, val_json_get_pathname)

    # Helper function to load or create dataset
    def load_or_create(cache_path, image_paths, mask_paths, json_path):
        if os.path.exists(cache_path):
            print(f"[INFO] Loading cached dataset from {cache_path}")
            inputs, labels = torch.load(cache_path)
            dataset = TensorDataset(inputs, labels)
        else:
            print(f"[INFO] Creating dataset and caching to {cache_path}")
            if cnn_type == "classifier":
                dataset = ThresholdDataSetWithMaskClassifier_Features(image_paths, mask_paths, unet_model, transform, json_path, device, data_input_type="pred_mask")
            else:
                dataset = ThresholdDataSetWithMask(image_paths, mask_paths, unet_model, transform, json_path, device)
            inputs = torch.stack([x for x, _ in dataset])
            labels = torch.tensor([y for _, y in dataset])
            torch.save((inputs, labels), cache_path)
            dataset = TensorDataset(inputs, labels)
        return dataset

    # Load or create datasets
    train_dataset = load_or_create(train_cache, image_paths, mask_paths, train_json_get_pathname)
    val_dataset = load_or_create(val_cache, val_image_paths, val_mask_paths, val_json_get_pathname)

    set_seed(seed)  # Redundant but safe before training
    # DataLoaders
    #train_loader = DataLoader(train_dataset, batch_size=threshold_batch_size, shuffle=True, num_workers=0, worker_init_fn=lambda _: np.random.seed(seed), generator=torch.Generator().manual_seed(seed))
    train_loader = make_loader(train_dataset, batch_size=threshold_batch_size, shuffle=True, seed=seed)
    #val_loader = DataLoader(val_dataset, batch_size=threshold_batch_size, shuffle=False, num_workers=0)
    val_loader = make_loader(val_dataset, batch_size=threshold_batch_size, shuffle=False, seed=seed)

    return {
        "train": train_loader,
        "val": val_loader
    }


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# ResNet10 Slimmed Down
class ResNet10_Small(nn.Module):
    def __init__(self, num_thresholds=20):
        super(ResNet10_Small, self).__init__()
        self.in_channels = 32  # Reduced from 64

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 1)
        self.layer2 = self._make_layer(64, 1, stride=2)
        self.layer3 = self._make_layer(128, 1, stride=2)
        self.layer4 = self._make_layer(256, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_thresholds)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)   # [B, 32, ...]
        x = self.layer2(x)   # [B, 64, ...]
        x = self.layer3(x)   # [B, 128, ...]
        x = self.layer4(x)   # [B, 256, ...]

        x = self.avgpool(x)  # [B, 256, 1, 1]
        x = torch.flatten(x, 1)
        return self.fc(x)


def visualize_prediction(pred_mask_tensor, pred_class, target_class):

    pred_mask_np = pred_mask_tensor.detach().cpu().numpy()

    plt.figure(figsize=(6, 4))

    if pred_mask_np.shape[0] == 1:  # Grayscale
        plt.imshow(pred_mask_np[0], cmap='gray')
    elif pred_mask_np.shape[0] == 3:  # RGB or 3-channel mask
        plt.imshow(np.transpose(pred_mask_np, (1, 2, 0)))
    else:  # Unexpected channel size
        plt.imshow(pred_mask_np[0], cmap='gray')

    plt.title(f"Predicted Class: {pred_class}, Target Class: {target_class}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compute_top_k_accuracy(preds, targets, k=3, num_classes=None):
    """
    Top-K accuracy: A prediction is correct if it's within ±(k-1) indices from the target index.
    
    Arguments:
    - preds: Tensor of shape (batch_size,) containing predicted class indices
    - targets: Tensor of shape (batch_size,) containing true class indices
    - k: integer, defines ±(k-1) range to consider
    - num_classes: total number of classes (required to prevent range overflow)
    """
    correct = 0
    batch_size = targets.size(0)

    if num_classes is None:
        num_classes = int(torch.max(targets).item() + 1)  # fallback

    for i in range(batch_size):
        pred_idx = preds[i].item()
        target_idx = targets[i].item()

        allowed_indices = set(range(
            max(0, target_idx - k + 1),
            min(num_classes, target_idx + k)
        ))

        if pred_idx in allowed_indices:
            correct += 1

    return correct / batch_size

def train_thresholdModelWithMaskClassifier(unet_model, transform, save_path="best_threshold_model.pth", do_training=False):
    device = getDevice()
    cnn_type= "classifier"

    train_json_filename = train_json_threshold_filename
    val_json_filename = val_json_threshold_filename
    threshold_loader = createThresholdDataLoaderWithMask(unet_model, train_json_filename, val_json_filename, transform, device)
    train_loader = threshold_loader['train']
    val_loader = threshold_loader['val']

    # Use ResNet-based classifier 
    model = ResNet10_Small(num_thresholds=num_thresholds).to(device)
    summary(model)

    if do_training == False:
       model.load_state_dict(torch.load('best_threshold_model.pth'))
       return model

    # Handle class imbalance if necessary
    class_weights = torch.ones(num_thresholds)  # Optional: use real weights from your data
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # If you want to continue training from the previous checkpoint (Pre-training)
    #checkpoint = torch.load("pre_train_best_threshold_model.pth")
    #model.load_state_dict(checkpoint)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for pred_masks, targets in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}"):
            pred_masks = pred_masks.to(device)
            targets = targets.to(device).long().squeeze()

            # === Preprocess predicted masks ===
            if pred_masks.size(1) == 1:
                pred_masks = pred_masks.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
            pred_masks = (pred_masks - 0.5) / 0.5  # Normalize to [-1, 1]
            '''
            # Ensure 1-channel (no repeat to RGB!)
            if pred_masks.size(1) != 1:
                pred_masks = pred_masks[:, :1, :, :]  # Just use the first channel if it's 3-channel

            # Normalize to [0, 1] if needed
            pred_masks = torch.clamp(pred_masks, 0, 1)
            # Normalize to [-1, 1]
            pred_masks = (pred_masks - 0.5) / 0.5
            '''

            optimizer.zero_grad()
            outputs = model(pred_masks)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # === Validation ===
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        close_val = 0
        total_val = 0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0

        with torch.no_grad():
            for i, (pred_masks, targets) in enumerate(val_loader):
                pred_masks = pred_masks.to(device)
                targets = targets.to(device).long().squeeze()

                if pred_masks.size(1) == 1:
                    pred_masks = pred_masks.repeat(1, 3, 1, 1)
                pred_masks = (pred_masks - 0.5) / 0.5
                '''
                # Ensure 1-channel (no repeat to RGB!)
                if pred_masks.size(1) != 1:
                    pred_masks = pred_masks[:, :1, :, :]  # Just use the first channel if it's 3-channel

                # Normalize to [0, 1] if needed
                pred_masks = torch.clamp(pred_masks, 0, 1)
                # Normalize to [-1, 1]
                pred_masks = (pred_masks - 0.5) / 0.5
                '''

                outputs = model(pred_masks)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                preds = outputs.argmax(dim=1)

                # Compute top-K correctness for the batch
                top1_correct += (preds == targets).sum().item()
                num_classes = model.fc.out_features if hasattr(model, "fc") else num_thresholds
                top3_correct += compute_top_k_accuracy(preds, targets, k=3, num_classes=num_classes) * targets.size(0)
                top5_correct += compute_top_k_accuracy(preds, targets, k=5, num_classes=num_classes) * targets.size(0)

                total_val += targets.size(0)

                #if epoch % 10 == 0 and i == 0:
                    #visualize_prediction(pred_masks[0], preds[0].item(), targets[0].item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_top1_accuracy = top1_correct / total_val
        val_top3_accuracy = top3_correct / total_val
        val_top5_accuracy = top5_correct / total_val

        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, "
      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
      f"Val Loss: {avg_val_loss:.4f}, "
      f"Top-1 Acc: {val_top1_accuracy:.4f}, "
      f"Top-3 Acc: {val_top3_accuracy:.4f}, "
      f"Top-5 Acc: {val_top5_accuracy:.4f}, "
      f"LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f">>> Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# Coarse Dropout implementation
class CoarseDropout(object):
    def __init__(self, p=0.5, max_height=16, max_width=16):
        self.p = p
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, image):
        if np.random.rand() > self.p:
            return image

        # Get the size of the image
        height, width = image.size[1], image.size[0]

        # Randomly pick the size of the dropout region
        drop_height = np.random.randint(1, self.max_height)
        drop_width = np.random.randint(1, self.max_width)

        # Randomly pick the location of the dropout region
        top = np.random.randint(0, height - drop_height)
        left = np.random.randint(0, width - drop_width)

        # Apply dropout
        #image[:, top:top + drop_height, left:left + drop_width] = 0
        # Convert to NumPy array
        image_np = np.array(image)

        # If grayscale image, convert to 3D by adding a channel dimension
        if image_np.ndim == 2:
            image_np = np.expand_dims(image_np, axis=-1)  # shape becomes (H, W, 1)

        # Apply the mask
        image_np[top:top + drop_height, left:left + drop_width, :] = 0

        # If array has shape (H, W, 1), squeeze the last dimension
        if image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = image_np.squeeze(axis=2)  # Shape becomes (H, W)
            image = Image.fromarray(image_np, mode='L')  # 'L' for grayscale

        # If RGB image (H, W, 3)
        elif image_np.ndim == 3 and image_np.shape[2] == 3:
            image = Image.fromarray(image_np)

        # If somehow still invalid, print shape to debug
        else:
            print("Invalid shape:", image_np.shape)
            raise ValueError("Unexpected image shape for conversion.")


        return image

# Synchronized data augmentation for both image and mask during training
class AugmentImageAndMask:
    def __init__(self, scale_factor=1.2, crop_size=(height, width)):
        self.scale_factor = scale_factor
        self.crop_size = crop_size
        self.resize = transforms.Resize((int(crop_size[0] * scale_factor), int(crop_size[1] * scale_factor)))
        self.random_crop = transforms.RandomCrop(crop_size)
    
    def __call__(self, image, mask):
        # Scaling
        image = self.resize(image)
        mask = self.resize(mask)
        
        # Cropping
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        # Convert to tensor
        image = transforms.functional.to_tensor(image)
        mask = transforms.functional.to_tensor(mask)
        
        return image, mask

# Synchronized Transformations for both image and mask including CoarseDropout
class SynchronizedTransforms:
    def __init__(self, resize=(height, width), hflip_prob=0.5, brightness=0.2, contrast=0.2, dropout_prob=0.3):
        self.resize = resize
        self.hflip_prob = hflip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.dropout = CoarseDropout(p=dropout_prob)  # Initialize CoarseDropout

    def __call__(self, image, mask):
        image = Ff.resize(image, self.resize)
        mask = Ff.resize(mask, self.resize)

        # Apply horizontal flip with probability
        if random.random() < self.hflip_prob:
            image = Ff.hflip(image)
            mask = Ff.hflip(mask)

        # Apply random brightness and contrast adjustment
        image = Ff.adjust_brightness(image, 1 + (random.random() - 0.5) * 2 * self.brightness)
        image = Ff.adjust_contrast(image, 1 + (random.random() - 0.5) * 2 * self.contrast)

        # Apply coarse dropout to image
        image = self.dropout(image)

        return image, mask


# Instantiate transform
sync_transform = SynchronizedTransforms()

class SoybeanPodDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, normalize_image=False, transform_seperate=True):
        '''
        Args Explanation:
            image_paths (list of str): List of file paths to input images
            mask_paths (list of str): List of file paths to corresponding binary masks (0 or 1)
            height (int): Height to resize images and masks
            width (int): Width to resize images and masks
            transform (callable, optional): Optional transform to be applied on a sample
            normalize_image (bool): Whether to normalize image to [0, 1] by dividing by 255.0
        '''
        self.image_paths = sorted(image_paths)  # Sort image file paths (done once)
        self.mask_paths = sorted(mask_paths)    # Sort label mask file paths (done once)
        self.transform = transform
        self.height = height
        self.width = width
        self.normalize_image = normalize_image
        self.transform_seperate = transform_seperate
         
        self.mask_trans = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
        ])
        
        # Define normalization transform
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load the image and mask
        image = Image.open(image_path).convert("RGB")  # Assuming RGB for the image
        mask = Image.open(mask_path).convert("L")     # 1-channel (grayscale) binary mask

        # Generate threshold target map from mask (numpy, then tensor)
        # Convert PIL image to tensor before squeezing
        mask_tensor = Ff.to_tensor(mask)  # shape: (1, H, W)


        # Convert images to numpy arrays and normalize
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        # Expand mask dimensions to match image (H, W, 1)
        mask = np.expand_dims(mask, axis=2)

        # Convert numpy arrays to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)    # (H, W, 1) -> (1, H, W)

        # Apply transformations if any
        if self.transform:
            if self.transform_seperate:
                image = self.transform(image)
                mask = self.transform(mask)
                #threshold_target = self.transform(threshold_target)
            else:
                #image, mask = self.transform(image, mask)
                image, mask, threshold_target = self.transform(image, mask, threshold_target)
            #mask = self.transform(mask)
                  
        # Normalize the image if required
        if self.normalize_image:
           image = self.normalize_transform(image)


        return image, mask, image_path
    

def save_augmented_images_and_masks(image_paths, mask_paths, output_image_dir, output_mask_dir, transform):
    # Clear and recreate directories
    if os.path.exists(output_image_dir):
        shutil.rmtree(output_image_dir)
    if os.path.exists(output_mask_dir):
        shutil.rmtree(output_mask_dir)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    sync_transform = SynchronizedTransforms()

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply augmentations (resize, flip, etc.)
        if transform:
            image_aug, mask_aug = transform(image, mask)
        else:
            image_aug, mask_aug = image, mask

        # Save augmented images and masks
        base_name = os.path.basename(image_path).split('.')[0]
        image_aug.save(os.path.join(output_image_dir, base_name + "_aug.png"))
        mask_aug.save(os.path.join(output_mask_dir, base_name + "_aug.png"))

        # Save original
        Image.open(image_path).resize((height, width)).save(os.path.join(output_image_dir, base_name + ".png"))
        Image.open(mask_path).resize((height, width)).save(os.path.join(output_mask_dir, base_name + ".png"))

    print("Augmented images and masks saved!")


def save_augmented_images_and_masks_multiple(image_paths, mask_paths, output_image_dir, output_mask_dir, transform, num_augments=5):
    # Clear and recreate directories
    if os.path.exists(output_image_dir):
        shutil.rmtree(output_image_dir)
    if os.path.exists(output_mask_dir):
        shutil.rmtree(output_mask_dir)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    for image_path, mask_path in zip(image_paths, mask_paths):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        base_name = os.path.basename(image_path).split('.')[0]

        for i in range(num_augments):
    
            if transform:
                image_aug, mask_aug = transform(image, mask)
            else:
               image_aug, mask_aug = image, mask

            # Save
            aug_image_name = f"{base_name}_aug_{i+1}.png"
            aug_mask_name = f"{base_name}_aug_{i+1}.png"
            image_aug.save(os.path.join(output_image_dir, aug_image_name))
            mask_aug.save(os.path.join(output_mask_dir, aug_mask_name))

        # Save original
        Image.open(image_path).resize((height, width)).save(os.path.join(output_image_dir, base_name + ".png"))
        Image.open(mask_path).resize((height, width)).save(os.path.join(output_mask_dir, base_name + ".png"))

    print(f"{num_augments} augmented images and masks saved per input image.")


def convert_numpy_array(array):
    array = array.squeeze()  # Now the shape is (3,)

    # Check if the values are in the range [0, 1], multiply by 255 if necessary
    if array.max() <= 1:
        array = (array * 255).astype(np.uint8)  # S

    return array


def get_image_filepaths(directory):
    '''Returns a list of filepaths for all images in the given directory.'''

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    image_paths = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and any(filename.endswith(ext) for ext in image_extensions):
            image_paths.append(filepath)

    return image_paths


def crop_tensor(tensor, top, left, height, width):
        '''
        Crops a tensor to the specified region.

        Args:
            tensor (torch.Tensor): The input tensor.
            top (int): The top coordinate of the crop box.
            left (int): The left coordinate of the crop box.
            height (int): The height of the crop box.
            width (int): The width of the crop box.

        Returns:
            torch.Tensor: The cropped tensor.
        '''
        return Ff.crop(tensor, top, left, height, width)


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

   
def numpy_to_image(array, image_file_name):
        image = Image.fromarray(array)
        # Save the image
        image.save(image_file_name)
        # Display the image
        #image.show()

def get_transform():
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToPILImage(),  # Convert tensor to PIL Image for transformation
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with probability of 0.5
        CoarseDropout(p=0.3, max_height=16, max_width=16),  # Coarse dropout with probability of 0.3
        transforms.ColorJitter(brightness=0.2),  # Random brightness adjustment
        transforms.ColorJitter(contrast=0.2),    # Random contrast adjustment
        transforms.ToTensor()  # Convert back to tensor
        ])
        
    return transform

def get_data_loaders(data_augment=True):
    # use the same transformations for train/val in this example
    trans_training = get_transform()
    trans_training = AugmentImageAndMask(scale_factor=1.2, crop_size=(height, width))

    trans = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToPILImage(),  # Convert tensor to PIL Image for transformation
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    mask_trans = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToPILImage(),  # Convert tensor to PIL Image for transformation
        transforms.ToTensor(),
        #transforms.Normalize([0.485], [0.229]) # imagenet   #one channel
    ])

    
    # Create dataset and dataloaders
    # Train loader
    # train_dir = "/path/to/my/images"
    image_paths = get_image_filepaths(train_dir)
    mask_paths = get_image_filepaths(train_mask_dir)

    val_image_paths = get_image_filepaths(val_dir)
    val_mask_paths = get_image_filepaths(val_mask_dir)

    # Save augmented images and masks
    #save_augmented_images_and_masks_multiple(image_paths, mask_paths, train_aug_dir, train_mask_aug_dir, sync_transform)  #(Comment this line if you have already saved the data augmentation images)
    #save_augmented_images_and_masks_multiple(val_image_paths, val_mask_paths, val_aug_dir, val_mask_aug_dir, sync_transform)  #(Comment this line if you have already saved the data augmentation images)

    # Load dataset from saved augmented images and masks
    #augmented_image_paths = [os.path.join(train_aug_dir, fname) for fname in os.listdir(train_aug_dir)]
    #augmented_mask_paths = [os.path.join(train_mask_aug_dir, fname) for fname in os.listdir(train_mask_aug_dir)]
    image_aug_paths = get_image_filepaths(train_aug_dir)
    mask_aug_paths = get_image_filepaths(train_mask_aug_dir)

   

    set_seed_1(seed1)  # Redundant but safe before training
    # DataLoaders

    train_dataset = SoybeanPodDataset(image_paths, mask_paths, transform=trans, transform_seperate=True)
    if data_augment:
       print("Using Data Augmentation for Training")
       train_dataset = SoybeanPodDataset(image_aug_paths, mask_aug_paths, transform=trans, transform_seperate=True)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #shuffle=True
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, worker_init_fn=lambda _: np.random.seed(seed1), generator=torch.Generator().manual_seed(seed1))
    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed1)


    # Create dataset and dataloaders
    # Validation loader
    val_dataset = SoybeanPodDataset(val_image_paths, val_mask_paths, transform=trans)
    data_augment = False
    if data_augment:
        val_image_aug_paths = get_image_filepaths(val_aug_dir)
        val_mask_aug_paths = get_image_filepaths(val_mask_aug_dir)
        val_dataset = SoybeanPodDataset(val_image_aug_paths, val_mask_aug_paths, transform=trans)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #shuffle=True
    val_loader   = make_loader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed1)


    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    return dataloaders


def compute_iou(preds, labels, eps=1e-6):
    """
    Compute IoU (Intersection over Union) per sample and average over the batch.
    Args:
        preds: tensor of shape (B, 1, H, W)
        labels: tensor of shape (B, 1, H, W)
    Returns:
        IoU: scalar tensor
    """
    preds = preds.view(preds.size(0), -1)
    labels = labels.view(labels.size(0), -1)

    intersection = (preds * labels).sum(dim=1)
    union = preds.sum(dim=1) + labels.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def compute_precision(preds, labels, eps=1e-6):
    """
    Compute precision per sample and average over the batch.
    Args:
        preds: tensor of shape (B, 1, H, W)
        labels: tensor of shape (B, 1, H, W)
    Returns:
        Precision: scalar tensor
    """
    preds = preds.view(preds.size(0), -1)
    labels = labels.view(labels.size(0), -1)

    true_positive = (preds * labels).sum(dim=1)
    false_positive = (preds * (1 - labels)).sum(dim=1)

    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    return precision.mean()

def get_dice_score(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    score = (2. * intersection + smooth) / (
        pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth
    )

    return score.mean()

def compute_iou_per_image(preds, labels):
    batch_size = preds.size(0)
    ious = []
    for i in range(batch_size):
        intersection = torch.sum(preds[i] * labels[i])
        union = torch.sum(preds[i]) + torch.sum(labels[i]) - intersection
        iou = intersection / union if union > 0 else torch.tensor(0.0, device=preds.device)
        ious.append(iou)
    return torch.stack(ious)

def compute_precision_per_image(preds, labels):
    batch_size = preds.size(0)
    precisions = []
    for i in range(batch_size):
        tp = torch.sum(preds[i] * labels[i])
        fp = torch.sum(preds[i] * (1 - labels[i]))
        precision = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=preds.device)
        precisions.append(precision)
    return torch.stack(precisions)

def get_dice_score_per_image(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).view(pred.size(0), -1).sum(dim=1)
    union = pred.view(pred.size(0), -1).sum(dim=1) + target.view(pred.size(0), -1).sum(dim=1)
    score = (2. * intersection + smooth) / (union + smooth)
    return score

def get_dice_loss(pred, target, smooth=1.):
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Flatten predictions and targets across height and width (dim=2 and dim=3)
    pred = pred.view(pred.size(0), -1)  # Flatten spatial dimensions (height * width)
    target = target.view(target.size(0), -1)  # Flatten spatial dimensions (height * width)

    # Calculate intersection and union
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    # Calculate Dice coefficient and then the loss
    dice_score = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice_score  # Dice loss is 1 minus Dice coefficient

    return loss.mean()  # Average the loss across the batch


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def compute_accuracy(pred, target, threshold=0.5):
    '''
    Calculate the accuracy of binary segmentation.

    Parameters:
    - pred: The model's output (logits), size [batch_size, 1, H, W]
    - target: The ground truth mask, size [batch_size, 1, H, W]
    - threshold: Threshold for binary classification, typically 0.5
    
    Returns:
    - accuracy: The accuracy as a percentage of correctly predicted pixels.
    '''

    # Apply sigmoid to get probabilities between 0 and 1
    pred = torch.sigmoid(pred)

    # Convert probabilities to binary predictions (0 or 1) using the threshold
    pred = (pred > threshold).float()

    # Compare predictions with the ground truth
    correct_pixels = (pred == target).sum()  # Count correct predictions
    total_pixels = target.numel()  # Total number of pixels in the target

    accuracy = (correct_pixels / total_pixels) * 100  # Accuracy as fraction of correct pixels
    return accuracy


def calc_loss(pred, target, metrics, bce_weight=0.5,
              threshold_pred=None, threshold_target=None,
              threshold_weight=25, entropy_weight=0.05):
    '''
    Computes combined BCE + Dice loss for segmentation,
    and optionally CrossEntropy + Entropy loss for threshold classification.
    
    Returns:
        total_loss (Tensor): Combined loss (segmentation + threshold classification).
    '''

    #device = getDevice()
    penalty_weight = 1e-3

    # --- Segmentation Loss (BCE + Dice) ---
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = get_dice_loss(pred, target)

    mask_loss = bce_weight * bce + (1 - bce_weight) * dice
    total_loss = mask_loss

    # Log segmentation metrics
    metrics['bce'] = metrics.get('bce', 0.0) + bce.item() * target.size(0)
    metrics['dice'] = metrics.get('dice', 0.0) + dice.item() * target.size(0)
    metrics['loss'] = metrics.get('loss', 0.0) + mask_loss.item() * target.size(0)

    # --- Threshold Classification Loss (optional) ---
    if threshold_pred is not None and threshold_target is not None:
        assert threshold_pred.dim() == 2 or threshold_pred.dim() == 1, "threshold_pred must be shape [N, 1] or [N]"
        assert threshold_target.dim() == 1 or threshold_target.dim() == 2, "threshold_target must be shape [N] or [N, 1]"
     
        reg_loss = F.smooth_l1_loss(threshold_pred.view(-1), threshold_target.view(-1))
           
        # Total threshold loss
        threshold_loss = reg_loss 
        total_loss += threshold_weight * threshold_loss

        # Logging
        metrics['threshold_regression'] = metrics.get('threshold_regression', 0.0) + reg_loss.item() * threshold_target.size(0)
        #metrics['threshold_entropy'] = metrics.get('threshold_entropy', 0.0) + entropy_loss.item() * threshold_target.size(0)
        metrics['threshold_loss'] = metrics.get('threshold_loss', 0.0) + threshold_loss.item() * threshold_target.size(0)

        print("threshold_pred (regression):", threshold_pred.view(-1))
        print("threshold_target (regression):", threshold_target.view(-1))
        print("regression_loss:", reg_loss)
        #print("entropy_loss:", entropy_loss)
        print("threshold_loss:", threshold_loss)
        print("total_loss:", total_loss)

    return total_loss


def compute_threshold_targets(inputs, labels, outputs):
                            device = getDevice()
                            threshold_classes = thresholds_tensor.to(device)
                            threshold_targets = []
                            for i in range(inputs.size(0)):
                                pred_mask = torch.sigmoid(outputs[i, 0])  # Convert logits to probabilities
                                gt_mask = labels[i, 0]
                                
                                best_thresh = 0.0
                                best_score = -1.0
                                best_idx = 0
                                
                                for idx, t in enumerate(threshold_classes):
                                    bin_mask = (pred_mask > t).float()
                                    intersection = (bin_mask * gt_mask).sum()
                                    union = bin_mask.sum() + gt_mask.sum()
                                    dice = (2. * intersection) / (union + 1e-6)
                                    
                                    if dice > best_score:
                                        best_score = dice
                                        best_thresh = t
                                        best_idx = idx
                                
                                threshold_targets.append(best_thresh)        

                            return threshold_targets
                                           


def train_model(model, optimizer, scheduler, num_epochs=25):
    dataloaders = get_data_loaders()
    device = getDevice()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    print(f"{'Epoch':<6}{'LR':<10}{'Train Loss':<12}{'Train Acc':<12}{'Val Loss':<12}{'Val Acc':<12}"
          f"{'Train Dice Loss':<18}{'Val Dice Loss':<18}{'Train Dice Score':<18}{'Val Dice Score':<18}")

    start = time.time()
    #threshold_classes = torch.linspace(0.4, 0.95, num_thresholds).to(device)
    threshold_classes = thresholds_tensor.to(device)
    

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']

        for phase in ['train', 'val']:
            if phase == 'train': 
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0
            dice_score_accum = 0.0
            iou_accum = 0.0
            precision_accum = 0.0
            num_of_epoch_batch = 0.0

            for inputs, labels, image_paths in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    # === Dynamically compute best threshold class index per sample ===
                    #threshold_targets = compute_threshold_targets(inputs, labels, outputs)
                    #threshold_targets = torch.tensor(threshold_targets, dtype=torch.float, device=device)

                    # Total loss: segmentation (BCE + Dice) + threshold classification
                    total_loss = calc_loss(
                        outputs,
                        labels,
                        metrics,
                        threshold_pred=None,
                        threshold_target=None
                    )

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # Dice Score computation for mask output
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float() #Instead of using fixed 0.5 thresholds, then use the adaptive learned output threshold. Check this and See if to compute the threshold per image
               
                # Assume output_threshold is [B, 10]         
                # Apply predicted threshold for each sample
                # preds = (probs > threshold_targets.view(-1, 1, 1, 1)).float()
                

                dice_scores = get_dice_score_per_image(preds, labels)
                iou_scores = compute_iou_per_image(preds, labels)
                precision_scores = compute_precision_per_image(preds, labels)

                dice_score_accum += dice_scores.sum().item()
                iou_accum += iou_scores.sum().item()
                precision_accum += precision_scores.sum().item()

                epoch_samples += preds.size(0)

            # Aggregate metrics
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_dice_loss = metrics['dice'] / epoch_samples
            epoch_threshold_loss = metrics.get('threshold_loss', 0.0) / epoch_samples
            epoch_dice_score = dice_score_accum / epoch_samples
            epoch_iou = iou_accum / epoch_samples
            epoch_precision = precision_accum / epoch_samples
            accuracy = compute_accuracy(outputs, labels)

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc_list.append(accuracy)
                train_dice_loss_list.append(epoch_dice_loss)
                train_dice_score_list.append(epoch_dice_score)
                train_iou_list.append(epoch_iou)
                train_precision_list.append(epoch_precision)
                train_threshold_loss_list.append(epoch_threshold_loss)
                scheduler.step()
            else:
                val_loss.append(epoch_loss)
                val_acc_list.append(accuracy)
                val_dice_loss_list.append(epoch_dice_loss)
                val_dice_score_list.append(epoch_dice_score)
                val_iou_list.append(epoch_iou)
                val_precision_list.append(epoch_precision)
                val_threshold_loss_list.append(epoch_threshold_loss)

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        print(f"{epoch+1:<6}{current_lr:<10.6f}{train_loss[-1]:<12.4f}{train_acc_list[-1]:<12.4f}"
              f"{val_loss[-1]:<12.4f}{val_acc_list[-1]:<12.4f}"
              f"{train_dice_loss_list[-1]:<18.4f}{val_dice_loss_list[-1]:<18.4f}"
              f"{train_dice_score_list[-1]:<18.4f}{val_dice_score_list[-1]:<18.4f}")

    print(f'Best Validation Loss: {best_loss:.6f}')
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "my_trained_model.pth")
    print(f"Training complete in {time.time() - start:.2f}s")

    return model


def train_and_validate_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=25):   
    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(20, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracy
    train_acc_list_cpu = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_acc_list]
    val_acc_list_cpu = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_acc_list]
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_acc_list_cpu, label='Train Accuracy')
    plt.plot(epochs, val_acc_list_cpu, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Dice Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_dice_loss_list, label='Train Dice Loss')
    plt.plot(epochs, val_dice_loss_list, label='Val Dice Loss')
    plt.title('Dice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Loss')
    plt.grid(True)
    plt.legend()

    # Dice Score
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_dice_score_list, label='Train Dice Score')
    plt.plot(epochs, val_dice_score_list, label='Val Dice Score')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.savefig("training_metrics.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Create a new figure for IoU and Precision with a grid layout
    plt.figure(figsize=(20, 10))

    # IoU Plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_iou_list, label='Train IoU', color='blue')
    plt.plot(epochs, val_iou_list, label='Val IoU', color='red')
    plt.title('IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()

    # Precision Plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_precision_list, label='Train Precision', color='green')
    plt.plot(epochs, val_precision_list, label='Val Precision', color='orange')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()

    # Save and display the plot
    plt.tight_layout()
    plt.savefig("iou_and_precision_plots.png")
    plt.savefig("iou_and_precision_plots.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return model


def analyze_pod_mask(binary_mask, image_name, binary_mask_th5, binary_mask_th9, binary_mask_th7, binary_mask_th9_5, binary_mask_th3):
    """
    Converts UNet output to a dot plot of soybean pods and returns the count and centroid locations.

    Parameters:
        model_output (torch.Tensor): UNet output tensor of shape (1, H, W) or (H, W)
        threshold (float): Threshold for binarizing the mask

    Returns:
        num_pods (int): Number of detected soybean pods
        centroids (list of tuples): List of (y, x) coordinates for each pod
    """
    '''
    # Step 1: Convert to NumPy array and threshold
    pred_mask = model_output[0, 0].squeeze().cpu().numpy()
    binary_mask = (pred_mask > threshold).astype(np.uint8)
    '''

    # Step 2: Connected component labeling
    labeled_mask, num_pods = label(binary_mask)

    print("Number of Pods", num_pods)

    # Connected component labeling for threshold 5
    labeled_mask_th5, num_pods_th5 = label(binary_mask_th5)

    # Connected component labeling for threshold 9
    labeled_mask_th9, num_pods_th9 = label(binary_mask_th9)

    # Connected component labeling for threshold 7
    labeled_mask_th7, num_pods_th7 = label(binary_mask_th7)

    # Connected component labeling for threshold 9_5
    labeled_mask_th9_5, num_pods_th9_5 = label(binary_mask_th9_5)

    # Connected component labeling for threshold 3
    labeled_mask_th3, num_pods_th3 = label(binary_mask_th3)

    # Step 3: Compute centroids
    centroids = center_of_mass(binary_mask, labeled_mask, range(1, num_pods + 1))

    return num_pods, centroids, num_pods_th5, num_pods_th9, num_pods_th7, num_pods_th9_5, num_pods_th3


def predict_binary_maskClassifier(unet_model, threshold_model, images, device, threshold_classes=None):
    unet_model.eval()
    threshold_model.eval()

    if threshold_classes is None:
        threshold_classes = thresholds_tensor.numpy()

    with torch.no_grad():
        # UNet forward pass
        logits = unet_model(images)                   # [B, 1, H, W]
        mask_prob = torch.sigmoid(logits)                # [B, 1, H, W]

        # Threshold model forward pass
        class_logits = threshold_model(mask_prob)        # [B, 18]
        probs = F.softmax(class_logits, dim=1)           # [B, 18]
        pred_class_indices = torch.argmax(probs, dim=1)  # [B]

        # Convert class indices to actual threshold values
        pred_threshold_values = torch.tensor(
            [threshold_classes[idx] for idx in pred_class_indices],
            dtype=torch.float32,
            device=device
        ).view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        # Apply thresholds
        binary_masks = (mask_prob > pred_threshold_values).float()

    return (
        mask_prob,                                     # Predicted probability masks
        binary_masks.cpu().numpy(),                   # Thresholded binary masks
        pred_threshold_values.squeeze().cpu().numpy(),# Predicted threshold values
        mask_prob.cpu().numpy(),                      # For visualization or analysis
        logits.cpu()                                  # Raw logits
    )

def predict_binary_maskClassifier_3channel(unet_model, threshold_model, images, device, threshold_classes=None):
    unet_model.eval()
    threshold_model.eval()

    if threshold_classes is None:
        threshold_classes = thresholds_tensor.numpy()

    with torch.no_grad():
        # === Step 1: Get predicted probability masks from UNet ===
        #logits = unet_model(images)                
        output = unet_model(images)                # [B, 1, H, W]
        if isinstance(output, tuple):
            logits = output[0]  # take the first element as main logits
        else:
            logits = output

        mask_prob = torch.sigmoid(logits)             # [B, 1, H, W]

        # === Step 2: Prepare mask input for threshold classifier ===
        if mask_prob.size(1) == 1:
            mask_prob_rgb = mask_prob.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        else:
            mask_prob_rgb = mask_prob

        mask_prob_rgb = (mask_prob_rgb - 0.5) / 0.5  # Normalize to [-1, 1]

        # === Step 3: Predict threshold class from mask ===
        class_logits = threshold_model(mask_prob_rgb)        # [B, num_thresholds]
        probs = F.softmax(class_logits, dim=1)               # [B, num_thresholds]
        pred_class_indices = torch.argmax(probs, dim=1)      # [B]

        # === Step 4: Map class index to actual threshold values ===
        pred_threshold_values = torch.tensor(
            [threshold_classes[idx] for idx in pred_class_indices],
            dtype=torch.float32,
            device=device
        ).view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        # === Step 5: Apply thresholds to generate binary masks ===
        binary_masks = (mask_prob > pred_threshold_values).float()
        binary_masks_th5 = (mask_prob > 0.5).float()
        binary_masks_th9 = (mask_prob > 0.9).float()
        binary_masks_th7 = (mask_prob > 0.7).float()
        binary_masks_th9_5 = (mask_prob > 0.95).float()
        binary_masks_th3 = (mask_prob > 0.3).float()


    return (
        mask_prob,                                       # Probability masks
        binary_masks.cpu().numpy(),                      # Final binary masks
        pred_threshold_values.squeeze().cpu().numpy(),   # Thresholds used
        mask_prob.cpu().numpy(),                         # Raw probs for analysis
        binary_masks_th5.cpu().numpy(),                  # binary masks at 0.5 threshold
        binary_masks_th9.cpu().numpy(),                  # binary masks at 0.9 threshold
        binary_masks_th7.cpu().numpy(),                  # binary masks at 0.7 threshold
        binary_masks_th9_5.cpu().numpy(),                  # binary masks at 0.95 threshold
        binary_masks_th3.cpu().numpy(),                  # binary masks at 0.3 threshold
        logits.cpu()                                     # Raw logits
    )


def calculate_test_metrics(test_json_save_name):
    # Load JSON file
    with open(test_json_save_name, "r") as f:
        data = json.load(f)

    empty_data = {}
    test_metrics_json_save_name = "test_metrics_json.json"
    with open(test_metrics_json_save_name, 'w') as f:
        json.dump(empty_data, f)

    threshold_id = "adaptive_threshold"
    threshold_ext = ""

    # Extract true and predicted pod counts
    y_true = []
    y_pred = []
    sum_absolute = 0
    count = 0

    '''
    for _, values in data.items():
        count += 1
        if count > 2:  # Only store in the lists the last 12 test images instead of 14
            y_true.append(values["pod_count"])
            y_pred.append(values["pred_mask_pod_count"])
    '''

    while count < 6:
        y_true = []
        y_pred = []
        sum_absolute = 0
        count = count+1; 

        if count == 2:
            threshold_id = "threshold_0.3"
            threshold_ext = "@_0.3th"    
        elif count == 3:
            threshold_id = "threshold_0.5"
            threshold_ext = "@_0.5th"    
        elif count == 4:
            threshold_id = "threshold_0.7"
            threshold_ext = "@_0.7th"    
        elif count == 5:
            threshold_id = "threshold_0.9"
            threshold_ext = "@_0.9th"  
        elif count == 6:
            threshold_id = "threshold_0.95"
            threshold_ext = "@_0.95th"   


        for _, values in data.items():
            pod_count = values["pod_count"]
            pred_mask_pod_count_th = values["pred_mask_pod_count"+threshold_ext]
            y_true.append(pod_count)
            y_pred.append( pred_mask_pod_count_th)
            sum_absolute = abs(sum_absolute) + abs(((pred_mask_pod_count_th - pod_count)/pod_count) * 100)

        print("")
        print("Th: "+threshold_id)
        print("y_true: ", y_true)
        print("y_pred: ", y_pred)

        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        rmae = mae / np.mean(np.abs(y_true))
        rrmse = rmse / np.mean(np.abs(y_true))
        mean_absolute_diff = sum_absolute / len(y_pred)
        accuracy = (1 - rmae) * 100

        with open(test_metrics_json_save_name, 'r') as f:
            all_data = json.load(f)

        all_data[threshold_id] = {
                "MAE" : round(mae, 4),
                "RMSE" : round(rmse, 4),
                "Accuracy" : round(accuracy, 4),
                "R^2" : round(r2, 4),
                "rMAE" : round(rmae, 4),
                "rRMSE" : round(rrmse, 4), 
                "MA_Diff" : round(mean_absolute_diff, 4) 
            }

        with open(test_metrics_json_save_name, "w") as f:
            json.dump(all_data, f, indent=4)

        # Print results
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"rMAE: {rmae:.4f}")
        print(f"rRMSE: {rrmse:.4f}")
        print(f"MA_Diff: {mean_absolute_diff:.4f}")


def soft_binarize(pred_mask, threshold, delta=0.05):
    threshold = threshold.view(-1, 1, 1, 1)  # reshape for broadcasting
    return torch.sigmoid((pred_mask - threshold) / delta)


def test_model(model):
    device = getDevice()
    # Load the saved model weights
    model.load_state_dict(torch.load('my_trained_model.pth'))

    # Perform model testing and prediction
    model.eval()  # Set model to the evaluation mode
    
    trans = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToPILImage(),  # Convert tensor to PIL Image for transformation
            transforms.ToTensor(),
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # imagenet
        ])

    generate_threshold_json_files = True

    if(generate_threshold_json_files):
        empty_data = {}
        os.makedirs(pod_json_save_path, exist_ok=True)
        train_json_save_path = os.path.join(pod_json_save_path, train_json_threshold_filename)
        val_json_save_path = os.path.join(pod_json_save_path, val_json_threshold_filename)
        #creating a json file for training info with a variable name as f
        with open(train_json_save_path, 'w') as f:
           json.dump(empty_data, f)

        empty_data = {}
        #creating a json file for validation info with a variable name as f
        with open(val_json_save_path, 'w') as f:
           json.dump(empty_data, f)

        train_threshold_data = BestThresholdPredMasks(train_aug_dir, train_mask_aug_dir, model, trans, pod_json_get_path, train_json_save_path, device) 
        val_threshold_data =   BestThresholdPredMasks(val_aug_dir, val_mask_aug_dir, model, trans, pod_json_get_path, val_json_save_path, device) 

    # This threshold_model is used to predict the best threshold
    threshold_model = train_thresholdModelWithMaskClassifier(model, trans, save_path="best_threshold_model.pth", do_training=True)

    # # Create another simulation dataset for test
    test_image_paths = get_image_filepaths(test_dir)
    test_mask_paths = get_image_filepaths(test_mask_dir)

    test_dataset = SoybeanPodDataset(test_image_paths, test_mask_paths, transform=trans)
    #test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    test_loader  = make_loader(test_dataset, batch_size=4, shuffle=False, seed=seed1)

    val_image_paths = get_image_filepaths(val_dir)
    val_mask_paths = get_image_filepaths(val_mask_dir)
    val_dataset = SoybeanPodDataset(val_image_paths, val_mask_paths, transform=trans)
    #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) #shuffle=True
    val_loader  = make_loader(val_dataset, batch_size=1, shuffle=False, seed=seed1)

    empty_data = {}
    test_json_save_name = "test_json_threshold.json"
    with open(test_json_save_name, 'w') as f:
        json.dump(empty_data, f)

    i = 0
    with torch.no_grad():
        for count, (inputs, labels, image_paths) in enumerate(test_loader, start=1):   # Replace test_loader with your test data
            inputs = inputs.to(device)

            predicted_mask, binary_mask, predicted_thresholds, mask_prob, binary_masks_th5, binary_masks_th9, binary_masks_th7, binary_masks_th9_5, binary_masks_th3, _ = predict_binary_maskClassifier_3channel(model, threshold_model, inputs, device)

                # Loop through each item in the batch
            for i in range(inputs.size(0)):
                    # Extract predicted binary mask for image i
                    #predicted_mask_numpy = binary_mask[i, 0]  # Already a NumPy array
                    predicted_mask_numpy = predicted_mask[i, 0].cpu().numpy()
                    predicted_mask_uint8 = (predicted_mask_numpy * 255).astype(np.uint8)
                    
                    # Convert to image
                    #output_mask_image = Image.fromarray(predicted_mask_uint8)
                    # Only proceed if it's 2D
                    if predicted_mask_uint8.ndim == 2:
                        output_mask_image = Image.fromarray(predicted_mask_uint8)
                    else:
                        raise ValueError(f"Expected 2D array, got shape {predicted_mask_uint8.shape}")

                    # Handle image path safely
                    image_name_without_ext = os.path.splitext(os.path.basename(image_paths[i]))[0]
                    mask_filename = image_name_without_ext + "_mask.jpg"

                    print(f"Saving mask for: {image_name_without_ext}")
                    output_mask_image.save(mask_filename)

                    # Optional: print threshold used
                    print(f"{mask_filename} | Threshold: {predicted_thresholds[i].item():.4f}")

                    # Optionally analyze pod mask
                    #pod_list_info = analyze_pod_mask(binary_mask[i:i+1], image_name_without_ext)
                    pod_list_info = analyze_pod_mask(binary_mask[i:i+1].squeeze(), image_name_without_ext, binary_masks_th5[i].squeeze(), binary_masks_th9[i].squeeze(), binary_masks_th7[i].squeeze(), binary_masks_th9_5[i].squeeze(), binary_masks_th3[i].squeeze())
                    pred_mask_pod_count = pod_list_info[0]

                    json_filename = os.path.join(pod_json_get_path, image_name_without_ext + ".json")
                    pod_coordinates = json_point_extract.extract_points_from_json(json_filename)
                    pod_count = len(pod_coordinates)

                    with open(test_json_save_name, 'r') as f:
                        all_data = json.load(f)

                    all_data[image_name_without_ext] = {
                         "pod_count" : int(pod_count),
                         "pred_mask_threshold" : float(predicted_thresholds[i]),
                         "pred_mask_pod_count" : int(pred_mask_pod_count),
                         "pred_mask_pod_count@_0.5th" : int(pod_list_info[2]),
                         "pred_mask_pod_count@_0.9th" : int(pod_list_info[3]),
                         "pred_mask_pod_count@_0.7th" : int(pod_list_info[4]),
                         "pred_mask_pod_count@_0.95th" : int(pod_list_info[5]),
                         "pred_mask_pod_count@_0.3th" : int(pod_list_info[6]),
                         "pred_mask_pod_count_difference" : float(((pred_mask_pod_count - pod_count)/pod_count) * 100),
                         "pred_mask_pod_count_difference@_0.5th" : float(((pod_list_info[2] - pod_count)/pod_count) * 100),
                         "pred_mask_pod_count_difference@_0.9th" : float(((pod_list_info[3] - pod_count)/pod_count) * 100),
                         "pred_mask_pod_count_difference@_0.7th" : float(((pod_list_info[4] - pod_count)/pod_count) * 100),
                         "pred_mask_pod_count_difference@_0.95th" : float(((pod_list_info[5] - pod_count)/pod_count) * 100),
                         "pred_mask_pod_count_difference@_0.3th" : float(((pod_list_info[6] - pod_count)/pod_count) * 100)
                    }


                    with open(test_json_save_name, "w") as f:
                        json.dump(all_data, f, indent=4)



            # Visualize the result
            plt.imshow(predicted_mask[0, 0].cpu().numpy(), cmap='gray')
            plt.savefig("plt_predicted_mask_" + str(count) + ".png", dpi=300, bbox_inches='tight')
            plt.savefig("plt_predicted_mask_" + str(count) + ".pdf", format="pdf", bbox_inches="tight")
            plt.show()    
          
            # Mean and std used for normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
    
            numpy_input_normalized = inputs[0].permute(1,2,0).cpu().numpy()
            numpy_input = numpy_input_normalized
            '''numpy_input = numpy_input_normalized * std + mean
            numpy_input = np.clip(numpy_input, 0, 1)'''
            numpy_label = labels[0].cpu().squeeze().numpy()
            numpy_predicted = predicted_mask[0,0].cpu().numpy()

            # Plot the images in a single row with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Show the test image     
            axes[0].imshow(numpy_input)
            axes[0].set_title("Test Image")

            # Show the label mask
            axes[1].imshow(numpy_label, cmap='gray')
            axes[1].set_title("Label Mask")

            # Show the predicted mask
            axes[2].imshow(numpy_predicted, cmap='gray')
            axes[2].set_title("Predicted Mask")

            # Display the plot
            plt.tight_layout()
            plt.savefig(f"plt_predicted_mask_{count}_comp.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"plt_predicted_mask_{count}_comp.pdf", format="pdf", bbox_inches="tight")
            plt.show()

        # Calculate the MSE, RMSE, R² Score, rMAE, rRMSE
        calculate_test_metrics(test_json_save_name)


def run(UNetLite):
    in_channels = 3
    num_class = 2
    num_out_channels = 1 # for binary segmentation, the output channel = 1
    num_epochs = 240
    image_size = (width, height)
    device = getDevice()

    model = UNetLite(in_channels, num_out_channels, image_size).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
    #optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0003, weight_decay=1e-3)

    # Step size is  The formula for step size (number of iterations per epoch) is:
    # Step_Size = Total_Training_Samples / Batch_Size
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
    #exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.7, patience=7, min_lr=1e-6, verbose=True)
    summary(model, input_size=(batch_size, in_channels, height, width))  #

    # Comment it if the model is already trained and weights saved in the file. We just load the saved model weights and do testing/prediction (The code i have shown below)
    model = train_and_validate_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
    test_model(model)
        