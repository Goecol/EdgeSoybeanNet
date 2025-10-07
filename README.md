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

EdgeSoybeanNet/
├── dataset/          # contains only test images
├── codes/            # all Python scripts (training, inference, utils, etc.)
├── trained_models/   # pre-trained model weights (.pth)
└── requirements.txt  # dependency list

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

### 🚀 Running the Model

**Option 1 — Direct Run**

```bash
python run_model.py
```

This will:

* Load the pretrained model from `/trained_models/`
* Run inference on all test images in `/dataset/test/`

---

### 🚀 Running EdgeSoybeanNet

**Option 2 — Run Directly from the Model File**

You can execute the model directly from the `UNetLite.py` file, which automatically imports and runs the engine:

```bash
python UNetLite.py
```

This script internally includes:

```python
import engine
engine.run(UNetLite)
```

Inside the `run()` function of `engine.py`, you can decide whether to **test**, **train**, or **evaluate** the model by modifying the code logic.
This gives you full control to switch between different modes or datasets easily.

*(You may later rename `engine.py` to `edge_runner.py` for clarity — especially if you plan to manage multiple models in the same repository.)*


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
