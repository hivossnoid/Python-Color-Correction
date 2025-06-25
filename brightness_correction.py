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

def detect_colors_lab(frame, person_mask, colors_to_detect, threshold=30):
    """
    Detect dominant colors in the person region using LAB distance.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mask_indices = np.where(person_mask > 0)
    detected = []

    output_frame = frame.copy()

    for color_name in colors_to_detect:
        if color_name not in COLOR_LAB_REFERENCES:
            continue
        
        target_lab = COLOR_LAB_REFERENCES[color_name]
        distances = np.linalg.norm(lab[mask_indices] - target_lab, axis=1)

        match_mask = np.zeros(person_mask.shape, dtype=np.uint8)
        match_pixels = (distances < threshold)

        # Only set matching pixels
        match_mask[mask_indices[0][match_pixels], mask_indices[1][match_pixels]] = 255

        # Clean up the mask
        cleaned_mask = clean_mask(match_mask)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                cv2.drawContours(output_frame, [contour], -1, DISPLAY_COLORS[color_name], 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(output_frame, (cx, cy), 5, DISPLAY_COLORS[color_name], -1)
                    cv2.putText(output_frame, color_name, (cx - 20, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DISPLAY_COLORS[color_name], 1)
                    detected.append({
                        'color': color_name,
                        'center': (cx, cy),
                        'area': area
                    })

    return output_frame, detected


def nothing(x):
    # This function solely exists for passing a value to the brightness tracker.
    pass
