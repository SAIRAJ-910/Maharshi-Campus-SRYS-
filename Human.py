import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_BLOCKTIME"] = "0"

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

image_path = "C:/Users/WELCOME/Desktop/INoVation/SREYAS Project3/people.webp"  

image = Image.open(image_path).convert("RGB")
transform = transforms.ToTensor()
img_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    predictions = model(img_tensor)

boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

threshold = 0.5
human_boxes = []
human_scores = []

for box, label, score in zip(boxes, labels, scores):
    if label == 1 and score >= threshold:
        human_boxes.append(box)
        human_scores.append(score)

human_count = len(human_boxes)

img_np = np.array(image)

for box, score in zip(human_boxes, human_scores):
    x1, y1, x2, y2 = box.int().tolist()
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img_np,
        f"Person {score:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_np)
plt.title(f"Detected {human_count} Humans")
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"Faster R-CNN detected {human_count} humans")

