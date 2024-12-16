"""
ocr_processor.py

This module provides the `TextExtractor` class for processing images and extracting textual data using PaddleOCR.

Key Features:
- Detect text in images and extract bounding box coordinates, recognized text, and confidence scores.
- Annotate images with bounding boxes around detected text.
- Format OCR results into structured data with coordinates and corresponding text.
- Organize detected text into rows based on vertical alignment for better usability.
- Save processed and annotated images for visualization.

The `TextExtractor` class is designed for seamless integration into text recognition pipelines and supports
custom configurations for OCR processing and output handling.
"""

from paddleocr import PaddleOCR
import cv2


class TextExtractor:
    """
    A class for processing images to detect and extract textual data using PaddleOCR.

    It includes methods for performing OCR, drawing bounding boxes around detected text,
    formatting and organizing text into structured rows, and saving processed images.
    """

    def __init__(self, enable_angle_cls: bool = True, language: str = 'en'):
        """
        Initializes the TextExtractor class with OCR configuration.

        :param enable_angle_cls: Whether to use angle classification during OCR.
        :param language: The language model to use for text recognition.
        """
        self.angle_classification = enable_angle_cls
        self.ocr_engine = PaddleOCR(use_angle_cls=self.angle_classification, lang=language)

    def extract_text(self, input_image):
        """
        Performs OCR on the given image.

        :param input_image: The image on which to perform OCR.
        """
        self.source_image = input_image
        ocr_results = self.ocr_engine.ocr(self.source_image, cls=self.angle_classification)

        self.bounding_boxes = [line[0] for line in ocr_results[0]]  # Extract bounding boxes
        self.detected_texts = [line[1][0] for line in ocr_results[0]]  # Extract recognized text
        self.confidence_scores = [line[1][1] for line in ocr_results[0]]  # Extract confidence scores

    def draw_bounding_box(self, bbox, color: tuple = (0, 255, 0), thickness: int = 2):
        """
        Draws a bounding box on the image.

        :param bbox: The coordinates of the bounding box.
        :param color: Color of the bounding box (default is green).
        :param thickness: Thickness of the bounding box lines.
        """
        # Convert coordinates to integers
        bbox = [list(map(int, point)) for point in bbox]

        # Draw lines to form a rectangle/polygon
        for i in range(len(bbox)):
            start = tuple(bbox[i])
            end = tuple(bbox[(i + 1) % len(bbox)])
            self.source_image = cv2.line(self.source_image, start, end, color, thickness)

        return self.source_image

    def annotate_image(self):
        """
        Draws all bounding boxes on the source image.

        :return: The annotated image.
        """
        for bbox in self.bounding_boxes:
            self.draw_bounding_box(bbox)
        return self.source_image

    def save_annotated_image(self, save_path: str = "annotated_output.png"):
        """
        Saves the annotated image to the specified path.

        :param save_path: File path to save the annotated image.
        """
        cv2.imwrite(save_path, self.source_image)

    def organize_text_data(self):
        """
        Formats OCR results into structured data with bounding box coordinates and text.

        :return: A list of dictionaries containing box coordinates and detected text.
        """
        structured_results = []
        for i in range(len(self.bounding_boxes)):
            structured_results.append({
                "coordinates": self.bounding_boxes[i],
                "text": self.detected_texts[i]
            })
        return structured_results

    def group_text_into_rows(self):
        """
        Groups detected text into rows based on their vertical alignment.

        :return: A list of lists, where each inner list contains text grouped by rows.
        """
        ocr_results = self.organize_text_data()

        # Calculate the average y-coordinates for each bounding box
        for item in ocr_results:
            y_coords = [point[1] for point in item['coordinates']]
            x_coords = [point[0] for point in item['coordinates']]
            item['avg_y'] = sum(y_coords) / len(y_coords)
            item['avg_x'] = sum(x_coords) / len(x_coords)

        # Sort results by vertical alignment (avg_y)
        ocr_results.sort(key=lambda item: item['avg_y'])

        # Group texts into rows
        rows = []
        current_row = []
        current_avg_y = ocr_results[0]['avg_y']
        row_threshold = 10  # Y-coordinate threshold for grouping

        for item in ocr_results:
            if abs(item['avg_y'] - current_avg_y) < row_threshold:
                current_row.append(item)
            else:
                # Sort the row horizontally (avg_x) and append it
                current_row.sort(key=lambda entry: entry['avg_x'])
                rows.append([entry['text'] for entry in current_row])
                current_row = [item]
                current_avg_y = item['avg_y']

        # Append the last row if it exists
        if current_row:
            current_row.sort(key=lambda entry: entry['avg_x'])
            rows.append([entry['text'] for entry in current_row])

        return rows
