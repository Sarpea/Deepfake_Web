
from transformers import AutoProcessor, AutoModelForVideoClassification
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import io
import base64

# Import the model definition
from modeling import DeepFakeDetector, load_model

# Constants
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def preprocess_frame(frame):
    # Convert to PIL Image if it's a numpy array
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    
    # Resize
    frame = frame.resize((im_size, im_size))
    
    # Convert to tensor
    frame = np.array(frame).astype(np.float32) / 255.0
    frame = (frame - np.array(mean)) / np.array(std)
    frame = frame.transpose(2, 0, 1)  # HWC -> CHW
    frame = torch.tensor(frame, dtype=torch.float32)
    
    return frame

def inference(model_inputs):
    # Load the model
    model = load_model()
    
    # Process inputs
    if "frames" in model_inputs:
        # Process frames from base64
        frames = []
        for frame_b64 in model_inputs["frames"]:
            img_data = base64.b64decode(frame_b64)
            frame = Image.open(io.BytesIO(img_data))
            frame = np.array(frame)
            frames.append(preprocess_frame(frame))
        
        # Stack frames
        frames = torch.stack(frames)
        frames = frames.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            _, outputs = model(frames)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # Get prediction (0: fake, 1: real)
            prediction = int(np.argmax(probs))
            confidence = float(probs[prediction]) * 100
            
            return {
                "prediction": "REAL" if prediction == 1 else "FAKE",
                "confidence": round(confidence, 1),
                "prediction_code": prediction
            }
    
    # Default response if no valid input
    return {"error": "Invalid input format"}
