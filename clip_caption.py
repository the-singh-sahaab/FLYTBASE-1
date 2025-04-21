import cv2
import torch
from PIL import Image
import numpy as np
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from ultralytics import YOLO
from description_generator_clip import generate_description

class CLIPLLMCaptioner:
    def __init__(self, gpt2_model="gpt2", device=None, yolo_weights="yolov8s.pt"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.yolo = YOLO(yolo_weights)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model).to(self.device)
        self.class_names = self.yolo.model.names

    def detect_objects_clip(self, frame):
        """
        Use CLIP zero-shot classification for object detection.
        Returns detected class names and their scores.
        """
        img = Image.fromarray(frame[..., ::-1])
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.class_names]).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity.cpu().numpy()[0]
        detected = [(self.class_names[i], float(probs[i])) for i in np.argsort(probs)[::-1] if probs[i] > 0.10]
        detected_classes = [c for c, p in detected]
        detected_scores = [p for c, p in detected]
        return detected_classes, detected, detected_scores

    def detect_objects_yolo(self, frame):
        """
        Use YOLOv8 for object detection.
        Returns detected class names and their scores.
        """
        results = self.yolo(frame)
        classes = []
        scores = []
        for r in results:
            for cls, conf in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                classes.append(self.class_names[int(cls)])
                scores.append(float(conf))
        from collections import defaultdict
        class_score = defaultdict(float)
        for c, s in zip(classes, scores):
            class_score[c] = max(class_score[c], s)
        detected = sorted(class_score.items(), key=lambda x: -x[1])
        detected_classes = [c for c, s in detected]
        detected_scores = [s for c, s in detected]
        return detected_classes, detected, detected_scores

    def draw_clip_annotations(self, frame, detected):
        """
        Draw top detected class names and scores on the frame.
        """
        annotated = frame.copy()
        for idx, (label, score) in enumerate(detected[:5]):
            y = 30 + idx * 30
            cv2.putText(
                annotated,
                f"{label} ({score:.2f})",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        return annotated

    def draw_yolo_annotations(self, frame, results):
        """
        Draw YOLO bounding boxes and class names on the frame.
        """
        return results[0].plot()

    def generate_caption(self, detections):
        """
        Use GPT-2 to generate a caption from detected object list.
        """
        if not detections:
            return "No objects detected in the scene."
        prompt = (
            "Given the following objects detected in a scene: "
            + ", ".join(detections)
            + ". Write a natural, concise English sentence describing the scene:\n"
        )
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.gpt2.generate(
            input_ids,
            max_length=input_ids.shape[1] + 20,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        caption = generated[len(prompt):].strip().split("\n")[0]
        if not caption or caption.lower().startswith("given"):
            caption = "Detected " + ", ".join(detections) + " in the scene."
        return caption

    def caption(self, frame):
        detected_classes, detected, detected_scores = self.detect_objects_yolo(frame)
        results = self.yolo(frame)
        annotated = self.draw_yolo_annotations(frame, results)
        caption = self.generate_caption(detected_classes)
        return caption, annotated

class CLIPCaptioner:
    """
    Captioner using OpenAI CLIP for zero-shot object/concept detection and Phi-2 for natural language scene description.
    - Uses the full set of CLIP's pretrained class labels (e.g., COCO or ImageNet).
    - Returns a one-sentence description and an annotated frame.
    """

    def __init__(self, device="cpu", class_labels=None):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        # Use a large set of class labels (COCO 80 classes by default)
        if class_labels is not None:
            self.labels = class_labels
        else:
            # Use CLIP's built-in ImageNet class labels
            import clip
            self.labels = clip.available_models()["ViT-B/32"]["imagenet_classes"]

    def detect_objects_clip(self, frame, threshold=0.12, top_k=10):
        """
        Use CLIP zero-shot classification for object detection.
        Returns detected class names and their scores.
        """
        img = Image.fromarray(frame[..., ::-1])
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.labels]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity.cpu().numpy()[0]
        # Get top_k classes above threshold
        indices = np.argsort(probs)[::-1]
        detected = [(self.labels[i], float(probs[i])) for i in indices if probs[i] > threshold][:top_k]
        detected_classes = [c for c, p in detected]
        detected_scores = [p for c, p in detected]
        return detected_classes, detected, detected_scores

    def draw_clip_annotations(self, frame, detected):
        """
        Draw top detected class names and scores on the frame.
        """
        annotated = frame.copy()
        for idx, (label, score) in enumerate(detected[:5]):
            y = 30 + idx * 30
            cv2.putText(
                annotated,
                f"{label} ({score:.2f})",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        return annotated

    def caption(self, frame):
        """
        Returns (description, annotated_frame) for the input frame.
        """
        detected_classes, detected, detected_scores = self.detect_objects_clip(frame)
        description = generate_description(detected_classes)
        annotated = self.draw_clip_annotations(frame, detected)
        return description, annotated
