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

def resplit_words_to_boxes(words, boxes):
    """
    Assigns a list of OCR'd words to a list of bounding boxes based on relative widths.

    Args:
        words (list): A list of recognized word strings.
        boxes (list): A list of original Crafter boxes (polygons), sorted by x-coordinate.

    Returns:
        list: A list of dictionaries, each containing a box and its assigned text.
    """
    if not words or not boxes:
        return []

    # Calculate total word characters (including spaces) and total width of all boxes
    word_lengths = [len(w) for w in words]
    total_word_length = sum(word_lengths) + max(0, len(words) - 1)
    
    box_widths = [np.max(np.array(b)[:, 0]) - np.min(np.array(b)[:, 0]) for b in boxes]
    total_box_width = sum(box_widths)

    if total_box_width == 0:
        # Fallback for zero-width boxes: assign one word per box if possible
        return [{'box': box, 'text': words[i] if i < len(words) else ""} for i, box in enumerate(boxes)]

    assignments = []
    word_cursor = 0
    
    # Assign words to each box except the last one
    for i in range(len(boxes) - 1):
        box = boxes[i]
        box_width = box_widths[i]
        
        # Calculate the ideal number of characters this box should contain
        ideal_chars = (box_width / total_box_width) * total_word_length
        
        best_num_words = 0
        min_diff = float('inf')
        
        accumulated_chars = -1  # Start at -1 to account for no space before the first word
        # Find the best number of words to assign to this box
        for j in range(word_cursor, len(words)):
            accumulated_chars += word_lengths[j] + 1
            diff = abs(accumulated_chars - ideal_chars)
            
            if diff < min_diff:
                min_diff = diff
                best_num_words = j - word_cursor + 1
            else:
                # Difference started increasing, so the previous split was optimal
                break
        
        # Ensure at least one word is assigned if words are left
        if best_num_words == 0 and word_cursor < len(words):
            best_num_words = 1
            
        assigned_words = words[word_cursor : word_cursor + best_num_words]
        text = " ".join(assigned_words)
        assignments.append({'box': box, 'text': text})
        word_cursor += best_num_words
        
    # Assign all remaining words to the very last box
    if word_cursor < len(words) or len(assignments) < len(boxes):
        last_box = boxes[-1]
        remaining_words = words[word_cursor:]
        text = " ".join(remaining_words)
        assignments.append({'box': last_box, 'text': text})
        
    return assignments

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
    vertical_threshold = float(request.json.get('vertical_threshold', 0.5))
    max_new_tokens = int(request.json.get('max_new_tokens', 200))
    num_beams = int(request.json.get('num_beams', 5))
    early_stopping = request.json.get('early_stopping', True)
    no_repeat_ngram_size = int(request.json.get('no_repeat_ngram_size', 3))

    try:
        processed_image_cv = load_image(preprocessed_image_path)
        app.logger.info(f"Preprocessed image loaded from {preprocessed_image_path}")

        visualized_image = cv2.cvtColor(processed_image_cv, cv2.COLOR_GRAY2BGR) if len(processed_image_cv.shape) == 2 else processed_image_cv.copy()

        crafter = Crafter()
        app.logger.info("Crafter initialized for OCR.")
        
        app.logger.info("Performing text detection with Crafter...")
        prediction = crafter(preprocessed_image_path)
        app.logger.info(f"Crafter text detection complete. Found {len(prediction['boxes'])} boxes.")

        # --- Group word boxes into lines, preserving original boxes ---
        line_groups = []
        # Sort initial boxes by vertical position for stable, top-to-bottom processing
        sorted_boxes = sorted(prediction['boxes'], key=lambda box: np.min(np.array(box)[:, 1]))
        used = [False] * len(sorted_boxes)

        for i, box in enumerate(sorted_boxes):
            if used[i]: continue

            box_poly = np.array(box).astype(int)
            y_min, y_max = np.min(box_poly[:, 1]), np.max(box_poly[:, 1])
            
            current_line_boxes = [box]
            used[i] = True

            for j, other_box in enumerate(sorted_boxes):
                if used[j] or j == i: continue
                
                other_poly = np.array(other_box).astype(int)
                oy_min, oy_max = np.min(other_poly[:, 1]), np.max(other_poly[:, 1])

                box_height = y_max - y_min
                avg_height = (box_height + (oy_max - oy_min)) / 2
                
                # Check if vertical centers are closely aligned
                if avg_height > 0 and abs(((oy_min + oy_max) / 2) - ((y_min + y_max) / 2)) < vertical_threshold * avg_height:
                    current_line_boxes.append(other_box)
                    used[j] = True

            # Sort the boxes within the line horizontally
            current_line_boxes.sort(key=lambda b: np.min(np.array(b)[:, 0]))

            # Calculate the merged bounding box for the entire line
            all_points = np.vstack([np.array(b) for b in current_line_boxes])
            group_xmin, group_ymin = np.min(all_points, axis=0)
            group_xmax, group_ymax = np.max(all_points, axis=0)

            # Apply padding
            merged_box = [
                max(0, group_xmin - padding), max(0, group_ymin - padding),
                min(processed_image_cv.shape[1], group_xmax + padding), min(processed_image_cv.shape[0], group_ymax + padding)
            ]

            line_groups.append({"merged_box": merged_box, "constituent_boxes": current_line_boxes})
        
        app.logger.info(f"Grouped into {len(line_groups)} lines of text.")

        final_recognized_results = []
        for i, group in enumerate(line_groups):
            x_min, y_min, x_max, y_max = [int(c) for c in group['merged_box']]
            
            cropped_image = processed_image_cv[y_min:y_max, x_min:x_max]
            if cropped_image.size == 0: continue

            cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB) if len(cropped_image.shape) == 2 else cropped_image)

            # OCR with TrOCR on the entire line
            pixel_values = processor(images=cropped_image_pil, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens, num_beams=num_beams, early_stopping=early_stopping, no_repeat_ngram_size=no_repeat_ngram_size)
            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            app.logger.info(f"Line {i} OCR: '{recognized_text}'")

            # --- Resplit recognized words back to their original boxes ---
            words = recognized_text.split()
            constituent_boxes = group['constituent_boxes']
            
            if words:
                app.logger.info(f"Resplitting '{recognized_text}' among {len(constituent_boxes)} boxes for line {i}.")
                resplit_assignments = resplit_words_to_boxes(words, constituent_boxes)
                for assignment in resplit_assignments:
                    app.logger.info(f"  - Mapped text: '{assignment['text']}'")
                final_recognized_results.extend(resplit_assignments)
        
        # Prepare final data for JSON response and visualization
        response_texts_and_boxes = []
        for item in final_recognized_results:
            box_poly = np.array(item['box']).astype(int)
            x_min, y_min = np.min(box_poly, axis=0)
            x_max, y_max = np.max(box_poly, axis=0)
            
            response_texts_and_boxes.append({"box": [int(x_min), int(y_min), int(x_max), int(y_max)], "text": item['text']})
            cv2.rectangle(visualized_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        app.logger.info("All text recognition and re-splitting complete. Saving visualized image...")
        final_processed_filename = "final_processed_" + preprocessed_image_id
        temp_output_path = os.path.join(app.config['UPLOAD_FOLDER'], final_processed_filename)
        cv2.imwrite(temp_output_path, visualized_image)
        
        saved_path = None
        if save_preprocessed_image:
            persistent_filename = "processed_" + preprocessed_image_id
            saved_path = os.path.join(app.config['SAVE_PATH'], persistent_filename)
            cv2.imwrite(saved_path, visualized_image)
            app.logger.info(f"Visualized image saved persistently to {saved_path}")

        app.logger.info("Encoding visualized image to base64...")
        _, buffer = cv2.imencode('.png', visualized_image)
        final_processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        response_data = {
            "status": "success",
            "message": "OCR processed successfully",
            "preprocessed_image_base64": final_processed_image_base64,
            "recognized_texts_and_boxes": response_texts_and_boxes
        }
        if saved_path:
            response_data["saved_image_path"] = saved_path
            
        app.logger.info("Returning success response.")
        return jsonify(response_data)

    except Exception as e:
        app.logger.exception(f"OCR failed: {str(e)}")
        return jsonify({"error": f"OCR failed: {str(e)}"}), 500
    finally:
        if 'temp_output_path' in locals() and temp_output_path and os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            app.logger.info(f"Cleaned up temporary output file: {temp_output_path}")
        if preprocessed_image_path and os.path.exists(preprocessed_image_path):
            os.remove(preprocessed_image_path)
            app.logger.info(f"Cleaned up temporary preprocessed file: {preprocessed_image_path}")

if __name__ == '__main__':
    app.run(debug=True)