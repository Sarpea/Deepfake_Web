---
language: en
license: mit
tags:
  - deepfake
  - video-classification
  - pytorch
  - computer-vision
datasets:
  - custom
pipeline_tag: video-classification
widget:
  - example_title: Deepfake Detection
    example_input: "Upload a video to detect if it's real or fake"
---

# 🛡️ Advanced Deepfake Detection Engine

This repository contains a powerful model for detecting deepfake videos. It leverages a sophisticated hybrid architecture combining **ResNext50** for spatial feature extraction and an **LSTM** network for temporal analysis, allowing it to effectively discern authentic videos from synthetically manipulated ones.

---

## ✨ Core Features

* **High Accuracy:** Achieves **87% accuracy** on our internal test datasets, providing a reliable classification of video authenticity.
* **Hybrid Architecture:** The fusion of CNN (ResNext50) and RNN (LSTM) captures both the visual artifacts within frames and the temporal inconsistencies between them.
* **Built with PyTorch:** A flexible and powerful framework for deep learning.
* **Easy to Use:** Integrated with the Hugging Face `pipeline` for quick and straightforward inference.

## ⚙️ Model Specifications

| Parameter         | Details                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| **Model Type** | Video Classification                                                       |
| **Primary Task** | Binary Deepfake Detection                                                  |
| **Framework** | `PyTorch`                                                                  |
| **Training Data** | Trained on a diverse, custom-built dataset comprising real and deepfake videos. |
| **Output** | A binary classification (`real`/`fake`) coupled with a confidence score.   |
| **Architecture** | `ResNext50` + `LSTM`                                                       |



## 🧠 Under the Hood: Model Architecture
The model's effectiveness stems from its two-stage architecture designed to mimic how a human might spot inconsistencies in a video:

* **Spatial Feature Extractor (ResNext50):** For each frame in a video sequence, a pre-trained **ResNext50** network acts as a powerful backbone. It analyzes the image to extract a rich set of spatial features, identifying subtle visual artifacts, unnatural textures, or lighting inconsistencies that are common hallmarks of deepfakes.

* **Temporal Sequence Analyzer (LSTM):** The sequence of feature vectors extracted by ResNext50 is then fed into a **Long Short-Term Memory** (LSTM) network. The LSTM's role is to analyze the temporal flow of these features over time. It excels at detecting unnatural transitions, flickering, or inconsistent movements between frames—anomalies that are often missed by single-frame analysis.

This combination allows the model to build a comprehensive understanding of both the look and the flow of a video to make a final, robust prediction.

## ⚠️ Limitations & Important Considerations
* **Focus on Human Faces:** The model was predominantly trained on datasets featuring human faces. Its performance is optimized for this domain and may be less reliable on videos without clear facial shots.

* **Video Quality Dependency:** Performance can be affected by video quality. Heavy compression, low resolution, or excessive motion blur might obscure the subtle artifacts the model looks for.

* **Optimal Sequence Length:** The model was designed and trained on video clips of **20 frames**. While it can process videos of other lengths, its optimal performance is achieved on sequences of this approximate duration.

* **Ethical Use:** This tool is intended for beneficial applications, such as combating misinformation and verifying media authenticity. Users are responsible for deploying it ethically and in accordance with privacy regulations.

## 🚀 Quickstart: Usage with `transformers`

Get started in just a few lines of Python. The easiest way to use this model is with the `video-classification` pipeline from the `transformers` library.

```python
# Ensure you have the necessary libraries installed
# pip install transformers torch torchvision

from transformers import pipeline

# 1. Initialize the video classification pipeline with our model
print("Loading the Deepfake Detection model...")
detector = pipeline("video-classification", model="Naman712/Deep-fake-detection")

# 2. Provide the path to your video file
video_path = "path/to/your/video.mp4"
print(f"Analyzing video: {video_path}...")

# 3. Get the prediction
result = detector(video_path)

# 4. Print the result
# The output will be a list of dictionaries with labels ('real' or 'fake') and scores.
print("Analysis Complete!")
print(result)


