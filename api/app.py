from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
import os
import uuid
import cv2
import base64
import numpy as np
from PIL import Image
from crafter import Crafter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging
import queue
import json
import time

from image_preprocessing import load_image, grayscale_image, denoise_image, binarize_image, deskew_image

app = Flask(__name__, static_folder='../frontend', static_url_path='/')
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['SAVE_PATH'] = 'preprocessed_images' # Directory to save preprocessed images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAVE_PATH'], exist_ok=True)

# Configure logging to capture messages
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

handler = QueueHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

PADDING = 5  # pixels of padding around each bounding box


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/progress_stream')
def progress_stream():
    def generate():
        while True:
            if not log_queue.empty():
                message = log_queue.get()
                yield f"data: {json.dumps({'log': message})}\n\n"
            time.sleep(0.1) # Small delay to prevent busy-waiting
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/upload_and_preprocess', methods=['POST'])
def upload_and_preprocess():
    app.logger.info("Received request to upload and preprocess image.")
    if 'image' not in request.files:
        app.logger.error("No image file provided in the request.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        app.logger.error("No selected image file.")
        return jsonify({"error": "No selected image file"}), 400

    # Get preprocessing parameters from request.form or use defaults
    enable_preprocessing = request.form.get('enable_preprocessing', 'true').lower() == 'true'

    temp_input_path = None
    preprocessed_temp_path = None

    if file:
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        temp_input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(temp_input_path)
        app.logger.info(f"Original image saved temporarily to {temp_input_path}")

        try:
            image = load_image(temp_input_path)
            app.logger.info("Image loaded for preprocessing.")

            processed_image_cv = image
            if enable_preprocessing:
                app.logger.info("Preprocessing enabled. Applying steps...")
                denoised = denoise_image(image)
                deskewed = deskew_image(denoised)
                grayscale = grayscale_image(deskewed)
                binarized = binarize_image(grayscale)
                processed_image_cv = binarized
                app.logger.info("Preprocessing complete.")
            else:
                app.logger.info("Preprocessing disabled.")

            # Save the preprocessed image temporarily for subsequent OCR
            preprocessed_unique_filename = "preprocessed_" + unique_filename
            preprocessed_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_unique_filename)
            cv2.imwrite(preprocessed_temp_path, processed_image_cv)
            app.logger.info(f"Preprocessed image saved temporarily to {preprocessed_temp_path}")

            # Encode the intermediate preprocessed image to base64 for immediate display
            _, buffer_intermediate = cv2.imencode('.png', processed_image_cv)
            intermediate_preprocessed_image_base64 = base64.b64encode(buffer_intermediate).decode('utf-8')
            app.logger.info("Intermediate preprocessed image encoded to base64.")

            response_data = {
                "status": "success",
                "message": "Image uploaded and preprocessed successfully",
                "preprocessed_image_id": preprocessed_unique_filename, # ID to retrieve for OCR
                "intermediate_preprocessed_image_base64": intermediate_preprocessed_image_base64
            }
            app.logger.info("Returning preprocessing success response.")
            return jsonify(response_data)

        except Exception as e:
            app.logger.exception(f"Image preprocessing failed: {str(e)}")
            return jsonify({"error": f"Image preprocessing failed: {str(e)}"}), 500
        finally:
            # Clean up original uploaded file
            if temp_input_path and os.path.exists(temp_input_path):
                os.remove(temp_input_path)
                app.logger.info(f"Cleaned up temporary input file: {temp_input_path}")

@app.route('/perform_ocr', methods=['POST'])
def perform_ocr():
    app.logger.info("Received request to perform OCR.")
    preprocessed_image_id = request.json.get('preprocessed_image_id')
    if not preprocessed_image_id:
        app.logger.error("No preprocessed_image_id provided in the request.")
        return jsonify({"error": "No preprocessed_image_id provided"}), 400

    preprocessed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_image_id)
    if not os.path.exists(preprocessed_image_path):
        app.logger.error(f"Preprocessed image not found at {preprocessed_image_path}")
        return jsonify({"error": "Preprocessed image not found"}), 404

    # Get OCR parameters from request.json or use defaults
    save_preprocessed_image = request.json.get('save_preprocessed_image', True)
    padding = int(request.json.get('padding', PADDING))
    vertical_threshold = float(request.json.get('vertical_threshold', 0.3))
    max_new_tokens = int(request.json.get('max_new_tokens', 200))
    num_beams = int(request.json.get('num_beams', 5))
    early_stopping = request.json.get('early_stopping', True)
    no_repeat_ngram_size = int(request.json.get('no_repeat_ngram_size', 3))

    try:
        processed_image_cv = load_image(preprocessed_image_path)
        app.logger.info(f"Preprocessed image loaded from {preprocessed_image_path}")

        # Convert processed_image_cv to BGR if it's grayscale for visualization
        if len(processed_image_cv.shape) == 2 or (len(processed_image_cv.shape) == 3 and processed_image_cv.shape[2] == 1):
            visualized_image = cv2.cvtColor(processed_image_cv, cv2.COLOR_GRAY2BGR)
        else:
            visualized_image = processed_image_cv.copy()

        crafter = Crafter()  # Initialize Crafter
        app.logger.info("Crafter initialized for OCR.")
        
        app.logger.info("Performing text detection with Crafter...")
        prediction = crafter(preprocessed_image_path)
        app.logger.info(f"Crafter text detection complete. Found {len(prediction['boxes'])} boxes.")

        # --- Group word boxes into line boxes ---
        line_boxes = []
        used = [False] * len(prediction['boxes'])

        for i, box in enumerate(prediction['boxes']):
            if used[i]:
                continue

            # Current box coordinates
            box = np.array(box).astype(int)
            x_min = np.min(box[:, 0])
            y_min = np.min(box[:, 1])
            x_max = np.max(box[:, 0])
            y_max = np.max(box[:, 1])

            # Start a new line group
            group_xmin, group_ymin, group_xmax, group_ymax = x_min, y_min, x_max, y_max
            used[i] = True

            for j, other_box in enumerate(prediction['boxes']):
                if used[j] or j == i:
                    continue
                other = np.array(other_box).astype(int)
                ox_min, oy_min = np.min(other[:, 0]), np.min(other[:, 1])
                ox_max, oy_max = np.max(other[:, 0]), np.max(other[:, 1])

                # Check vertical overlap (same line)
                box_height = y_max - y_min
                other_height = oy_max - oy_min
                avg_height = (box_height + other_height) / 2

                # Allow overlap only if centers are close enough
                if abs((oy_min + oy_max) / 2 - (y_min + y_max) / 2) < vertical_threshold * avg_height:
                    group_xmin = min(group_xmin, ox_min)
                    group_ymin = min(group_ymin, oy_min)
                    group_xmax = max(group_xmax, ox_max)
                    group_ymax = max(group_ymax, oy_max)
                    used[j] = True

            # Apply padding
            group_xmin = max(0, group_xmin - padding)
            group_ymin = max(0, group_ymin - padding)
            group_xmax = min(processed_image_cv.shape[1], group_xmax + padding)
            group_ymax = min(processed_image_cv.shape[0], group_ymax + padding)

            line_boxes.append([group_xmin, group_ymin, group_xmax, group_ymax])

        app.logger.info(f"Grouped into {len(line_boxes)} line boxes.")

        recognized_texts_and_boxes = []

        for i, (x_min, y_min, x_max, y_max) in enumerate(line_boxes):
            cropped_image = processed_image_cv[y_min:y_max, x_min:x_max]
            if cropped_image.size == 0:
                app.logger.warning(f"Skipping empty cropped image for line box {i}.")
                continue

            if len(cropped_image.shape) == 2:
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

            cropped_image_pil = Image.fromarray(cropped_image)

            # OCR with TrOCR
            pixel_values = processor(images=cropped_image_pil, return_tensors="pt").pixel_values
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            app.logger.info(f"Line box {i}: Recognized text: '{recognized_text}'")

            recognized_texts_and_boxes.append({
                "box": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "text": recognized_text
            })

            # Draw rectangle for line boxes
            cv2.rectangle(visualized_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        app.logger.info("All text recognition complete. Saving visualized image...")
        # Save the visualized image temporarily
        final_processed_filename = "final_processed_" + preprocessed_image_id
        temp_output_path = os.path.join(app.config['UPLOAD_FOLDER'], final_processed_filename)
        cv2.imwrite(temp_output_path, visualized_image)
        app.logger.info(f"Visualized image saved temporarily to {temp_output_path}")

        # Conditionally save the visualized image to a persistent location
        saved_path = None
        if save_preprocessed_image:
            persistent_filename = "processed_" + preprocessed_image_id
            saved_path = os.path.join(app.config['SAVE_PATH'], persistent_filename)
            cv2.imwrite(saved_path, visualized_image)
            app.logger.info(f"Visualized image saved persistently to {saved_path}")

        app.logger.info("Encoding visualized image to base64...")
        # Encode the visualized image to base64
        _, buffer = cv2.imencode('.png', visualized_image)
        final_processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        app.logger.info("Image encoded to base64.")

        response_data = {
            "status": "success",
            "message": "OCR processed successfully",
            "preprocessed_image_base64": final_processed_image_base64,
            "recognized_texts_and_boxes": recognized_texts_and_boxes
        }
        if saved_path:
            response_data["saved_image_path"] = saved_path
        
        app.logger.info("Returning success response.")
        return jsonify(response_data)

    except FileNotFoundError as e:
        app.logger.error(f"File not found error: {str(e)}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.exception(f"OCR failed: {str(e)}")
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500
    finally:
        # Clean up temporary preprocessed file
        if preprocessed_image_path and os.path.exists(preprocessed_image_path):
            os.remove(preprocessed_image_path)
            app.logger.info(f"Cleaned up temporary preprocessed file: {preprocessed_image_path}")
        if temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            app.logger.info(f"Cleaned up temporary output file: {temp_output_path}")

    app.logger.error("Unknown error during file upload.")
    return jsonify({"error": "Unknown error during file upload"}), 500


if __name__ == '__main__':
    app.run(debug=True)
