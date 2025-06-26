import cv2
import numpy as np
from ultralytics import YOLO


def calculate_brightness(image):
    """Calculate the average brightness of the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,2])

# This will adjust the brightness from the detected person.
def adjust_brightness(image, brightness_value):
    """Adjust brightness of an image. brightness_value is from -100 to +100"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply brightness offset and clip to 0-255
    v = np.clip(v + brightness_value, 0, 255).astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    bright_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_img

def adjust_brightness_2(image, brightness_value):
    """
    Adjust brightness of an image. brightness_value is from -100 to +100.
    Uses linear scaling on the BGR image to increase or decrease brightness.
    """
    # Convert brightness from -100 to 100 into a scale factor between 0.0 and 2.0
    # Where 0 means completely black, 1 means original, and 2 means double brightness
    factor = (brightness_value + 100) / 100.0  # Converts to range [0.0, 2.0]
    
    # Scale the image and clip to valid range
    bright_img = np.clip(image * factor, 0, 255).astype(np.uint8)
    
    return bright_img

def auto_adjust_brightness_contrast(region, brightness_thresholds=(35, 50)):
    """
    Automatically adjusts brightness and contrast of the person region.
    
    Parameters:
        region (np.ndarray): The image region to adjust.
        brightness_thresholds (tuple): (low_brightness, optimal_brightness).
    
    Returns:
        adjusted_region (np.ndarray): Brightness and contrast corrected region.
        contrast_factor (float): Used contrast adjustment factor.
        brightness_before (float): Original brightness value.
    """
    brightness_before = calculate_brightness(region)

    # Initialize values
    brightness_offset = 0
    contrast_factor = 1.0

    # Determine brightness adjustment
    if brightness_before < brightness_thresholds[0]:
        brightness_offset = brightness_thresholds[1] - brightness_before
        contrast_factor = 1.6
    elif brightness_before < brightness_thresholds[1]:
        brightness_offset = brightness_thresholds[1] - brightness_before
        contrast_factor = 2.1
    else:
        brightness_offset = 0
        contrast_factor = 1.2  # Mild boost if brightness is already decent

    # Apply brightness and contrast
    brightened = adjust_brightness_2(region, int(brightness_offset))
    adjusted = adjust_contrast(brightened, contrast_factor)

    return adjusted, contrast_factor, brightness_before


def auto_adjust_contrast(image):
    """
    Automatically enhance contrast using CLAHE on the luminance channel (LAB space).
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge channels back
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr

def adjust_contrast(image, contrast_factor):
    """
    More visibly aggressive contrast adjustment using mean-centered scaling.
    """
    mean = np.mean(image)
    adjusted = (image - mean) * contrast_factor + mean
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


# For showing the output of the whole frame's brightness uncomment the code below to test.
# brightness = calculate_brightness(frame)
# cv2.putText(output_frame, f"Brightness: {brightness:.1f}", (10, 120),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def nothing(x):
    # This function solely exists for passing a value to the brightness tracker.
    pass
