from transformers import TrOCRProcessor , VisionEncoderDecoderModel
from PIL import Image
import requests
# id.jpg present in same directory
img = Image.open("image.png").convert("RGB")


processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
pixel_values = processor(images=img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)