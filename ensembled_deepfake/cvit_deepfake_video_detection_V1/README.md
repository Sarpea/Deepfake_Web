---
license: mit
language:
- en
metrics:
- accuracy
pipeline_tag: video-classification
tags:
- cvit
- deepfake-detection
- video-classification
- computer-vision
- vision-transformer
- binary-classification
---

# 🔍 Convolutional Vision Transformer (CViT) for Deepfake Detection

The **Convolutional Vision Transformer (CViT)** is a hybrid architecture combining the powerful spatial feature extraction capabilities of CNNs with the long-range dependency modeling of Vision Transformers (ViT). This model is purpose-built for detecting deepfake videos and is trained on DFDC.

---

## Model Architecture

### 1. Feature Learning (FL) Module - CNN Backbone
- Composed of **17 convolutional operations**.
- Unlike traditional VGG architectures, **FL focuses purely on feature extraction**, not classification.
- Accepts input of size **224 × 224 × 3 (RGB image)**.
- Outputs a **512 × 7 × 7** feature map.
- Contains **10.8 million learnable parameters**.

### 2. Vision Transformer (ViT) Module
- Receives CNN output (**512 × 7 × 7**) as its input.
- Converts the 7×7 patches into a **1 × 1024** sequence using linear embedding.
- Adds **positional embeddings** of shape **(2 × 1024)**.
- ViT Encoder uses:
  - **Multi-Head Self Attention (MSA)** with **8 attention heads**.
  - **MLP blocks** with:
    - First linear layer of **2048** units.
    - Final linear layer of **2 units** (binary classification: Fake / Real).
    - **ReLU activation** and **Softmax** for final probabilities.

---

## 🧪 Experimental Results

The CViT model was tested and evaluated across multiple deepfake datasets:

### 📊 FaceForensics++ Accuracy
| Dataset                               | Accuracy |
|--------------------------------------|----------|
| FaceForensics++ FaceSwap             | 69%      |
| FaceForensics++ DeepFakeDetection    | 91%      |
| FaceForensics++ Deepfake             | 93%      |
| FaceForensics++ FaceShifter          | 46%      |
| FaceForensics++ NeuralTextures       | 60%      |

> **Note**: Poor performance on the FaceShifter dataset is attributed to the model's difficulty in learning subtle visual artifacts.

---

### 🧪 DFDC Evaluation

| Model               | Validation | Test   |
|---------------------|------------|--------|
| **CViT**            | 87.25%     | **91.5%** |

- **Unseen DFDC test videos**: 400
- **Accuracy**: 91.5%
- **AUC Score**: 0.91

---

### 🧪 UADFV AUC Comparison

| Model         | Validation | FaceSwap | Face2Face |
|---------------|------------|----------|-----------|
| **CViT**      | **93.75%** | 69.69%   | 69.39%    |

---

## ⚙️ Training Configuration

- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam
- **Learning Rate**: 1e-4  
- **Weight Decay**: 1e-6  
- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate Scheduler**: Reduces LR by factor of 0.1 every 15 epochs
- **Normalization**:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

---

## 🧪 Inference Setup

- **Input**: 30 normalized facial images (per video)
- **Classification**:
  - Uses **log loss function** to compute confidence.
  - Output is a probability `y ∈ [0, 1]`
    - `0 < y < 0.5`: Real
    - `0.5 ≤ y ≤ 1`: Fake
- Log loss penalizes:
  - Random guesses
  - Confident but incorrect predictions

---

## 🛠 Inference Example

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="mhamza-007/cvit_deepfake_detection",
    filename="cvit2_deepfake_detection_ep_50.pth"
)

# Load model (example)
model = torch.load(model_path, map_location='cpu')
model.eval()