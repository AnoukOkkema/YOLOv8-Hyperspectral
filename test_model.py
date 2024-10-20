import numpy as np
import torch
from ultralytics import YOLO
import ultralytics
import os
from ultralytics import get_channels_from_yaml

# Dynamically adjust bands number (CHATGPT)?
model_path = "yolov8-cls-tst.yaml" # custom
bands = get_channels_from_yaml(model_path)

# Load the YOLOv8 model
model = YOLO(model_path, task="classify") 

# Determine if MPS (Apple's GPU support) is available and select the appropriate compute device
compute_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model.train(data="/Users/anouk/Documents/GitHub/YOLOv8-Hyperspectral/dataset", 
            epochs = 50, 
            batch = 32,
            device=str(compute_device),
            num_bands=bands
            )

best_model_path = "/Users/anouk/Documents/GitHub/YOLOv8-Hyperspectral/runs/classify/train/weights/best.pt"

model = YOLO(model=best_model_path)

output = model.val(
            split="test", data="/Users/anouk/Documents/GitHub/YOLOv8-Hyperspectral/dataset", 
            device=compute_device, 
            workers=0, 
            num_bands=10
        )

print(output.results_dict)
