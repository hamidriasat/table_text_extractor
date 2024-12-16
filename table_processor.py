"""
table_processor.py

This module provides a class for detecting and extracting tables from document images using a YOLO model.
It includes functionality to load a model, perform predictions, annotate images with bounding boxes,
save processed images, and export the model for iOS deployment.

The TableExtractor class is designed for streamlined table detection and processing workflows.
"""

import os
from ultralyticsplus import YOLO
import cv2
import torch


class TableExtractor:
    """
    A class for detecting and extracting tables from images using a YOLO model.

    This class provides functionality for loading a YOLO model, running predictions
    on input images, drawing bounding boxes around detected tables, and saving
    cropped table regions or processed images. It also supports exporting the model
    to CoreML and saving it for reuse.
    """

    def __init__(
            self,
            model_file: str,
            confidence_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            use_class_agnostic: bool = False,
            max_detections: int = 1000,
    ):
        """
        Initializes the TableExtractor class with model parameters.

        :param model_file: Path to the YOLO model.
        :param confidence_threshold: Confidence threshold for NMS.
        :param iou_threshold: IoU threshold for NMS.
        :param use_class_agnostic: Whether to use class-agnostic NMS.
        :param max_detections: Maximum number of detections allowed per image.
        """
        self.detector = YOLO(model_file)
        self.detector.overrides["conf"] = confidence_threshold
        self.detector.overrides["iou"] = iou_threshold
        self.detector.overrides["agnostic_nms"] = use_class_agnostic
        self.detector.overrides["max_det"] = max_detections
        self.detected_tables: list = []

    def load_and_predict(self, image_path: str) -> list:
        """
        Loads an image and performs table detection.

        :param image_path: Path to the input image.
        :return: List of detection results.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at the specified path: {image_path}")

        input_image = cv2.imread(image_path)
        detections = self.detector.predict(input_image)
        self.source_image = input_image  # Store for further processing
        return detections

    def annotate_image(self, detections: list) -> None:
        """
        Draws bounding boxes around detected tables on the image and crops the tables.

        :param detections: The results from the model's prediction method.
        """
        for detection in detections:
            for table_box in detection.boxes:
                bbox_coords = (table_box.xyxy).tolist()[0]
                x_min, y_min, x_max, y_max = map(int, bbox_coords)

                cv2.rectangle(self.source_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                cropped_table = self.source_image[y_min:y_max, x_min:x_max]
                self.detected_tables.append(cropped_table)

    def save_processed_image(self, save_path: str = "processed_output.png") -> None:
        """
        Saves the image with annotations to a specified path.

        :param save_path: Path for saving the processed image.
        """
        cv2.imwrite(save_path, self.source_image)

    def execute(self, img_path: str) -> list:
        """
        Runs the full detection and processing pipeline.

        :param img_path: Path to the input image.
        :return: List of cropped table images.
        """
        results = self.load_and_predict(img_path)
        self.annotate_image(results)
        self.save_processed_image()
        return self.detected_tables

    def reset(self) -> None:
        """
        Clears the stored table data for new detections.
        """
        self.detected_tables.clear()

    def convert_to_coreml(self) -> None:
        """
        Exports the model to CoreML format for iOS deployment.
        """
        self.detector.export(format="coreml")

    def persist_model(self, save_file: str = "model_checkpoint.pt") -> None:
        """
        Saves the model to a file for future use.

        :param save_file: Path to save the model checkpoint.
        """
        torch.save(self.detector, save_file)
