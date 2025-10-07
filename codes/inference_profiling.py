import time
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import UNetLite_test as UNetLite



def getDevice():
    device = None
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_on_gpu = torch.cuda.is_available()
    #train_on_gpu = torch.backends.mps.is_available()
    train_on_gpu = False

    if not train_on_gpu:
        print('CUDA/MPS is not available.  Inference on CPU ...')
        device = torch.device("cpu")
    else:
        print('CUDA/MPS is available!  Inference on GPU ...')
        #device = torch.device("mps")
        device = torch.device("cuda")
    
    return device


# ----------------------------
# 1. Load your model
# ----------------------------
# Example: UNet in PyTorch

device = getDevice()

# 1.a Create the model
model = UNetLite.UNetLite(in_channels=3, out_channels=1)  # adjust channels if needed
model.to(device)

# 1.b Load weights
state_dict = torch.load("my_trained_model.pth", map_location=device)
model.load_state_dict(state_dict)

# 1.c Set to evaluation mode
model.eval()


# ----------------------------
# 2. Define preprocessing
# ----------------------------
transform = T.Compose([
    T.Resize((572, 572)),          # Resize to model's expected input
    T.ToTensor(),                  # Convert to tensor
    T.Normalize(mean=[0.5], std=[0.5])  # Adjust as needed for your model
])

# Example image path
image_path = "DJI_0626_0.bmp"

# ----------------------------
# 3. Profiling each stage
# ----------------------------

# Preprocessing
start_pre = time.time()
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)
end_pre = time.time()

# Forward pass (inference)
start_inf = time.time()
with torch.no_grad():
    output = model(input_tensor)
end_inf = time.time()

# Postprocessing (e.g., thresholding to binary mask)
start_post = time.time()
output_np = output.squeeze().cpu().numpy()
binary_mask = (output_np > 0.5).astype(np.uint8)  # threshold = 0.5
end_post = time.time()

# ----------------------------
# 4. Results
# ----------------------------
print(f"Preprocessing time: {(end_pre - start_pre) * 1000:.2f} ms")
print(f"Inference time:     {(end_inf - start_inf) * 1000:.2f} ms")
print(f"Postprocessing time:{(end_post - start_post) * 1000:.2f} ms")
print(f"Total time:         {(end_post - start_pre) * 1000:.2f} ms")