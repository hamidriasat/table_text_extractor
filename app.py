"""
image_processing_server.py

This script implements a Flask-based server that processes document images to extract tabular data.
The server uses the `TableExtractor` class for table detection and the `TextExtractor` class for text extraction
from cropped table images.

Key Features:
- Initializes and loads machine learning models for table detection.
- Accepts image file via POST requests for processing.
- Detects tables within images and extracts their contents into structured JSON responses.
- Supports saving annotated images with detected tables and extracted text.

Routes:
- /extract_text_from_image [POST]: Accepts JSON input with an `image` and returns tabular data extracted from the image.

Dependencies:
- Flask for API development.
- Logging for debugging and server monitoring.
- Custom classes `TableExtractor` and `TextExtractor` for AI-based table detection and text recognition.

Ensure that the `table_detector.py` and `text_extractor.py` modules are updated and available.
"""
from flask import Flask, request, jsonify
import os
import logging
from table_processor import TableExtractor
from ocr_processor import TextExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


def load_initial_setup():
    """
    Function that runs when the server starts
    """
    print("Server is starting up...")

    print("Initializing table processor...")

    global table_detector
    # Initializing table processor
    table_detector = TableExtractor(
        model_file="foduucom/table-detection-and-extraction",
    )

    print("Models loaded successfully")



@app.route("/extract_text_from_image", methods=["POST"])
def extract_text_from_image():
    try:
        # Access the global processor
        global table_detector

        # Check if processor is initialized
        if table_detector is None:
            return (
                jsonify({"error": "Table processor not initialized", "status": "error"}),
                500,
            )

        # Check if the post request has the file
        if "image" not in request.files:
            return jsonify({"error": "No image file provided in the variable 'image'.", "status": "error"}), 400

        image_file = request.files["image"]

        # Check if the file is a valid image (you can add further validation as needed)
        if not image_file:
            return jsonify({"error": "Invalid image file", "status": "error"}), 400

        # Save the uploaded image temporarily
        base_folder = "./images"
        image_path = os.path.join(base_folder, image_file.filename)
        image_file.save(image_path)

        # Perform table detection
        tables_cropped_images = table_detector.execute(image_path)

        # Initialize OCR processor
        ocr_processor = TextExtractor()  # Updated class name

        tables_data = []
        for table_index, table_image in enumerate(tables_cropped_images):
            ocr_processor.extract_text(table_image)
            ocr_processor.annotate_image()
            ocr_processor.save_annotated_image()
            table_rows = ocr_processor.group_text_into_rows()

            table_no = table_index + 1
            rows_data = []
            for row_index, row in enumerate(table_rows):
                row_no = row_index + 1
                rows_data.append({"row_no": row_no, "data": row})

            tables_data.append({"table_no": table_no, "rows": rows_data})

        table_detector.reset()
        return jsonify(tables_data)

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500



if __name__ == "__main__":
    load_initial_setup()  # Load

    app.run(debug=True, host="0.0.0.0", port=8000)  # Start the server
