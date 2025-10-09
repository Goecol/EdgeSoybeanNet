# 🌿 EdgeSoybeanNet

**An Edge-AI Framework for Real-Time, High-Accuracy Soybean Pod Counting**

---

### 🧠 Overview

**EdgeSoybeanNet** is an AI-powered framework designed for **real-time, high-accuracy soybean pod counting**.
Optimized for **edge deployment**, it enables efficient field-level analysis without heavy computational resources.

This repository provides:

* Implementation of the EdgeSoybeanNet model
* Pre-trained model weights
* Evaluation and inference scripts
* Configuration examples for deployment

---

### 📦 Repository Structure

```
EdgeSoybeanNet/
├── dataset/              # Contains only test images
│   └── ...               # (e.g., sample1.jpg, sample2.jpg)
│
├── json/                 # Contains JSON annotation files for test images only
│   └── ...               # (e.g., sample1.json, sample2.json)
│
├── codes/                # All Python scripts (training, inference, utils, etc.)
│   ├── Pre_processing/   # Codes for the preprocessing stage (Stage 1 of the framework)
│   │   └── ...   
│   ├── UNetLite.py       # Lightweight UNet model definition and visualization
│   ├── engine.py         # Core training loop and evaluation functions
│
├── results/              # Sample outputs from each stage of the framework
│   └── ...    
│
├── trained_models/       # Pre-trained model weights (.pth files)
│   └── ...
│
└── requirements.txt      # Python dependencies
```

> **Note:**
> The `json/` directory includes **only the JSON annotation files for the test dataset** used in evaluation.
> For access to the complete set of JSON files (training, validation, and testing) used in this research,
> please refer to the **EdgeSoybeanNet Research Paper** for dataset access links and preparation instructions.

---

### 🧰 Installation

**1. Clone the repository**

```bash
git clone https://github.com/Goecol/EdgeSoybeanNet.git
cd EdgeSoybeanNet
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv env
source env/bin/activate    # On macOS/Linux
env\Scripts\activate       # On Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

### 📂 Dataset Information

The repository includes **only the test dataset** for demonstration.
The **training and validation datasets** are large and not uploaded here.
Please refer to the **EdgeSoybeanNet Research Paper** for dataset access links and preparation instructions.

---

### 🚀 Running EdgeSoybeanNet

You can run the EdgeSoybeanNet model directly from the main model file:

```bash
python UNetLite.py
```

This script automatically imports the engine module and executes the model using:

```python
import engine
engine.run(UNetLite)
```

Inside the `run()` function of **`engine.py`**, you can specify whether to **train**, **test**, or **evaluate** the model by adjusting the logic in that function.
This gives you full flexibility to switch between different modes or datasets.

*(Optionally, you can rename `engine.py` to `edge_runner.py` later for better clarity when managing multiple models.)*

---

### ⚙️ Edge Deployment

The framework is optimized for **low-power edge devices** such as:

* NVIDIA Jetson Nano / Xavier
* Raspberry Pi (with Coral Edge TPU)
* Mobile devices using TFLite

Deployment configurations and scripts will be added soon.

---

### 🔄 Future Updates

More procedures, training instructions, and optimization details for **EdgeAI deployment** will be added soon.
Stay tuned for updates on:

* Full dataset preparation guide
* Model quantization and pruning scripts
* Edge deployment benchmarks

---

### 📜 Citation

If you use this work, please cite the **EdgeSoybeanNet** research paper.

---

### 🔗 Repository

GitHub: [https://github.com/Goecol/EdgeSoybeanNet](https://github.com/Goecol/EdgeSoybeanNet)
