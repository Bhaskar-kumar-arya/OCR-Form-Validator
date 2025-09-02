import cv2
import numpy as np

def load_image(image_path):
    """
    Loads an image from a given path using OpenCV.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def grayscale_image(image):
    """
    Converts an image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise_image(image):
    """
    Applies a denoising filter to the image (e.g., fastNlMeansDenoisingColored).
    """
    # Parameters for fastNlMeansDenoisingColored:
    # h: filter strength for luminance component. Higher h value removes more noise, but also removes more image detail. (default 3)
    # hColor: filter strength for color components. (default 3)
    # templateWindowSize: size in pixels of the template patch that is used to compute weights. Should be odd. (default 7)
    # searchWindowSize: size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. (default 21)
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def binarize_image(image):
    """
    Applies Otsu's binarization to the grayscale image.
    Assumes input image is grayscale.
    """
    if len(image.shape) == 3: # Check if image is not grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def deskew_image(image):
    """
    Corrects the skew of an image using OpenCV.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert the image (text white on black background)
    gray = cv2.bitwise_not(gray)

    # Threshold the image to get a binary image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))

    # Get the minimum area bounding rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Ensure the angle is in the range [-45, 45]
    # The angle returned by minAreaRect is in the range [-90, 0)
    # If the width is less than the height, it means the rectangle is "portrait"
    # and the angle needs to be adjusted by 90 degrees.
    if rect[1][0] < rect[1][1]: # if width < height
        angle = 90 + angle

    # Adjust the angle to be positive if needed, and within a reasonable range
    if angle > 45:
        angle = angle - 90
    elif angle < -45:
        angle = angle + 90

    # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def rotate_image(image, angle):
    """
    Rotates an image by a given angle.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def resize_image(image, scale_percent):
    """
    Resizes an image by a given percentage.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized
