from flask import Flask, request, jsonify
import os
import uuid
import cv2
import base64
import numpy as np
from PIL import Image
from crafter import Crafter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from image_preprocessing import load_image, grayscale_image, denoise_image, binarize_image, deskew_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['SAVE_PREPROCESSED_IMAGE'] = True  # Set to True to save, False to not save
app.config['SAVE_PATH'] = 'preprocessed_images' # Directory to save preprocessed images
app.config['ENABLE_PREPROCESSING'] = False # Set to True to enable preprocessing, False to disable
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAVE_PATH'], exist_ok=True)

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")


@app.route('/process_image', methods=['POST'])
def process_image():
    app.logger.info("Received request to process image.")
    if 'image' not in request.files:
        app.logger.error("No image file provided in the request.")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        app.logger.error("No selected image file.")
        return jsonify({"error": "No selected image file"}), 400

    temp_input_path = None
    temp_output_path = None

    if file:
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        temp_input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(temp_input_path)
        app.logger.info(f"Image saved temporarily to {temp_input_path}")

        try:
            crafter = Crafter()  # Initialize Crafter
            app.logger.info("Crafter initialized.")
            # Load the image
            image = load_image(temp_input_path)
            app.logger.info("Image loaded.")

            processed_image_cv = image
            if app.config['ENABLE_PREPROCESSING']:
                app.logger.info("Preprocessing enabled. Applying steps...")
                # Apply preprocessing steps
                denoised = denoise_image(image)
                deskewed = deskew_image(denoised) # Apply deskewing after denoising
                grayscale = grayscale_image(deskewed)
                binarized = binarize_image(grayscale)
                processed_image_cv = binarized
                app.logger.info("Preprocessing complete.")

            # Convert processed_image_cv to BGR if it's grayscale, so we can draw colored boxes
            if len(processed_image_cv.shape) == 2 or (len(processed_image_cv.shape) == 3 and processed_image_cv.shape[2] == 1):
                visualized_image = cv2.cvtColor(processed_image_cv, cv2.COLOR_GRAY2BGR)
            else:
                visualized_image = processed_image_cv.copy()

            # Perform text detection with Crafter
            app.logger.info("Performing text detection with Crafter...")
            if app.config['ENABLE_PREPROCESSING']:
                processed_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "processed_" + unique_filename)
                cv2.imwrite(processed_temp_path, processed_image_cv)
                prediction = crafter(processed_temp_path)
                os.remove(processed_temp_path)  # Clean up the temporary processed image
            else:
                prediction = crafter(temp_input_path)
            app.logger.info(f"Crafter text detection complete. Found {len(prediction['boxes'])} boxes.")

            recognized_texts_and_boxes = []

            for i, box in enumerate(prediction['boxes']):
                # Ensure integer numpy array
                box = np.array(box).astype(int)

                # Get bounding rectangle around the polygon
                x_min = np.min(box[:, 0])
                y_min = np.min(box[:, 1])
                x_max = np.max(box[:, 0])
                y_max = np.max(box[:, 1])

                # Crop safely within bounds
                cropped_image = processed_image_cv[y_min:y_max, x_min:x_max]
                if cropped_image.size == 0:
                    app.logger.warning(f"Skipping empty cropped image for box {i}.")
                    continue

                # Convert cropped_image to 3 channels if it's grayscale
                if len(cropped_image.shape) == 2 or (len(cropped_image.shape) == 3 and cropped_image.shape[2] == 1):
                    cropped_image_display = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
                else:
                    cropped_image_display = cropped_image.copy()

                cropped_image_pil = Image.fromarray(cropped_image_display)

                # OCR with TrOCR
                pixel_values = processor(images=cropped_image_pil, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values, max_new_tokens=200, num_beams=5, early_stopping=True, no_repeat_ngram_size=3)
                recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                app.logger.info(f"Box {i}: Recognized text: '{recognized_text}'")

                recognized_texts_and_boxes.append({
                    "box": box.tolist(),
                    "text": recognized_text
                })

                # Draw red bounding polygon
                pts = box.reshape((-1, 1, 2))
                cv2.polylines(visualized_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            app.logger.info("All text recognition complete. Saving visualized image...")
            # Save the visualized image temporarily
            preprocessed_filename = "preprocessed_" + unique_filename
            temp_output_path = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename)
            cv2.imwrite(temp_output_path, visualized_image)
            app.logger.info(f"Visualized image saved temporarily to {temp_output_path}")

            # Conditionally save the visualized image to a persistent location
            saved_path = None
            if app.config['SAVE_PREPROCESSED_IMAGE']:
                persistent_filename = "processed_" + unique_filename
                saved_path = os.path.join(app.config['SAVE_PATH'], persistent_filename)
                cv2.imwrite(saved_path, visualized_image)
                app.logger.info(f"Visualized image saved persistently to {saved_path}")

            app.logger.info("Encoding visualized image to base64...")
            # Encode the visualized image to base64
            _, buffer = cv2.imencode('.png', visualized_image)
            preprocessed_image_base64 = base64.b64encode(buffer).decode('utf-8')
            app.logger.info("Image encoded to base64.")

            response_data = {
                "status": "success",
                "message": "Image processed successfully",
                "preprocessed_image_base64": preprocessed_image_base64,
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
            app.logger.exception(f"Image processing failed: {str(e)}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500
        finally:
            # Clean up temporary files
            if temp_input_path and os.path.exists(temp_input_path):
                os.remove(temp_input_path)
                app.logger.info(f"Cleaned up temporary input file: {temp_input_path}")
            if temp_output_path and os.path.exists(temp_output_path):
                os.remove(temp_output_path)
                app.logger.info(f"Cleaned up temporary output file: {temp_output_path}")

    app.logger.error("Unknown error during file upload.")
    return jsonify({"error": "Unknown error during file upload"}), 500


if __name__ == '__main__':
    app.run(debug=True)
