from ultralytics import YOLO
from PIL import Image
import os
import logging

class FracturePredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("FracturePredictor")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")
        return YOLO(self.model_path)

    def predict(self, image: Image.Image, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Runs inference on the provided image.
        """
        self.logger.info(f"Running inference with conf={conf_threshold}, iou={iou_threshold}")
        results = self.model(image, conf=conf_threshold, iou=iou_threshold)
        return results

    def get_fracture_counts(self, results):
        """
        Returns a dictionary of detected classes and their counts.
        """
        counts = {}
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = names[class_id]
                counts[class_name] = counts.get(class_name, 0) + 1
        return counts
