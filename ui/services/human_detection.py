import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

class HumanDetector:
    def __init__(self, model_path="models/person_detection/efficientdet_lite0.tflite"):
        self.model = Interpreter(model_path=model_path, num_threads=2)
        self.model.allocate_tensors()
        self.input = self.model.get_input_details()[0]
        self.outputs = self.model.get_output_details()

    def detect(self, frame, threshold=0.5):
        height, width = frame.shape[:2]
        image = cv2.cvtColor(cv2.resize(frame, (320, 320)), cv2.COLOR_BGR2RGB)
        self.model.set_tensor(self.input["index"], image[None, ...])
        self.model.invoke()

        boxes = self.model.get_tensor(self.outputs[0]["index"])[0]
        classes = self.model.get_tensor(self.outputs[1]["index"])[0]
        scores = self.model.get_tensor(self.outputs[2]["index"])[0]
        count = int(self.model.get_tensor(self.outputs[3]["index"])[0])

        humans = []
        for i in range(min(count, len(scores))):
            if int(classes[i]) != 0 or float(scores[i]) < threshold:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = max(0, min(width - 1, int(xmin * width)))
            y1 = max(0, min(height - 1, int(ymin * height)))
            x2 = max(0, min(width - 1, int(xmax * width)))
            y2 = max(0, min(height - 1, int(ymax * height)))
            if x2 > x1 and y2 > y1:
                humans.append({
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "confidence": float(scores[i]),
                    "name": "Human"
                })
        return humans



def is_human_track(track_box, human_detections, threshold=0.30):
    """Check how much of a motion box overlaps a detected human."""
    x, y, w, h = track_box
    if w <= 0 or h <= 0:
        return False

    for detection in human_detections:
        hx, hy, hw, hh = detection["box"]
        ix = max(0, min(x + w, hx + hw) - max(x, hx))
        iy = max(0, min(y + h, hy + hh) - max(y, hy))
        overlap = (ix * iy) / (w * h)

        if overlap >= threshold:
            return True

    return False
