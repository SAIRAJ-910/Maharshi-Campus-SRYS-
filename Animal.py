import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_BLOCKTIME"] = "0"

import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Full COCO classes (complete list needed)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def faster_rcnn_inference(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)

    return {'boxes': predictions[0]['boxes'], 'scores': predictions[0]['scores'], 'labels': predictions[0]['labels']}

def show_results(image_path, results, threshold=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes, scores, labels = results['boxes'], results['scores'], results['labels']
    mask = scores > threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    for box, score, label in zip(boxes, scores, labels):
        box = box.cpu().numpy()  # ✅ FIXED: No [0] indexing
        x1, y1, x2, y2 = map(int, box)
        class_name = COCO_CLASSES[int(label)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {score:.2f}',
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# ✅ FIXED: Same image path for both!
results = faster_rcnn_inference("C:/Users/WELCOME/Desktop/INoVation/SREYAS Project3/cat.jpg")
show_results("C:/Users/WELCOME/Desktop/INoVation/SREYAS Project3/cat.jpg", results)

# ✅ FIXED Video loop
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb_frame)
    img_tensor = transforms.ToTensor()(pil_frame).unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)

    boxes, scores, labels = predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']

    for box, score, label in zip(boxes, scores, labels):  # ✅ All detections
        if score > 0.5:
            box = box.cpu().numpy()  # ✅ FIXED: Process ALL boxes
            x1, y1, x2, y2 = map(int, box)
            class_name = COCO_CLASSES[int(label)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}',
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
