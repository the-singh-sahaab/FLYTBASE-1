from ultralytics import YOLO
from collections import Counter
from description_generator_yolo import generate_description
import torch

class YOLOCaptioner:
    def __init__(self, weights_path="yolov8s.pt"):
        self.model = YOLO(weights_path)

    def caption(self, frame):
        results = self.model(frame)
        classes = []
        for r in results:
            classes += [self.model.model.names[int(cls)] for cls in r.boxes.cls.cpu().numpy()]
        # Use GPT-2 to generate a natural-language description
        description = generate_description(classes)
        annotated = results[0].plot()
        return description, annotated
