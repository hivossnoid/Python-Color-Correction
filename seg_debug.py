import cv2
import numpy as np
from ultralytics import YOLO

# Dictionary of HSV ranges for common colors
COLOR_RANGES = {
    'red': [
        ([0, 120, 70], [10, 255, 255]),      # Lower red range
        ([170, 120, 70], [179, 255, 255])    # Upper red range
    ],
    'orange': ([10, 120, 120], [25, 255, 255]),
    'yellow': ([20, 100, 100], [40, 255, 255]),
    'green': ([40, 80, 80], [80, 255, 255]),
    'blue': ([100, 80, 80], [140, 255, 255]),
    'indigo': ([130, 80, 80], [145, 255, 255]),
    'violet': ([145, 80, 80], [160, 255, 255]),
    'white': ([0, 0, 200], [179, 50, 255]),
    'gray': ([0, 0, 100], [179, 50, 200]),
    'black': ([0, 0, 0], [179, 255, 50]),
    'brown': ([10, 100, 50], [20, 255, 150])
}

# Different colors for visualization (BGR format)
DISPLAY_COLORS = {
    'red': (0, 0, 255),
    'orange': (0, 165, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'indigo': (130, 0, 75),
    'violet': (238, 130, 238),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'black': (0, 0, 0),
    'brown': (42, 42, 165)
}

def calculate_brightness(image):
    """Calculate the average brightness of the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:,:,2])

def clean_mask(mask):
    """Apply morphological operations to clean the mask"""
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    return cleaned

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

def detect_colors_in_segmented_person(frame, colors_to_detect, person_mask, min_area=500):
    """Detects colors within the segmented person area following the outline"""
    # Pre-process frame
    blurred = cv2.GaussianBlur(frame, (7,7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    output_frame = frame.copy()
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    all_objects = []
    
    # Draw the person contour (thicker outline)
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 3)
    
    for color_name in colors_to_detect:
        if color_name not in COLOR_RANGES:
            print(f"Warning: Color {color_name} not defined. Skipping.")
            continue
        
        color_ranges = COLOR_RANGES[color_name]
        display_color = DISPLAY_COLORS[color_name]
        
        # Get color mask
        if color_name == 'red':
            lower1, upper1 = color_ranges[0]
            lower2, upper2 = color_ranges[1]
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            color_mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = color_ranges
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Apply person mask to restrict color detection to person area
        color_mask = cv2.bitwise_and(color_mask, color_mask, mask=person_mask)
        
        # Clean the mask
        cleaned_mask = clean_mask(color_mask)
        
        # Add to combined mask
        combined_mask = cv2.bitwise_or(combined_mask, cleaned_mask)
        
        # Find contours of the color regions within the person
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the color regions following the person's outline
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Only proceed if the area meets the minimum size
            if area > min_area:
                # Draw the contour of the color region
                cv2.drawContours(output_frame, [contour], -1, display_color, 2)
                
                # Calculate the centroid of the color region
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    cv2.circle(output_frame, (center_x, center_y), 5, display_color, -1)
                    
                    # Display color name near the centroid
                    cv2.putText(output_frame, color_name, (center_x - 20, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 1)
                    
                    all_objects.append({
                        'color': color_name,
                        'center': (center_x, center_y),
                        'area': area
                    })
    
    # Create result image with only detected objects
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    return output_frame, result, all_objects


def nothing(x):
    # This function solely exists for passing a value to the brightness tracker.
    pass

def main():
    # Load YOLOv8n-seg modelx
    model = YOLO('yolov8n-seg.pt')
    
    cap = cv2.VideoCapture(0)
    # Specify which colors you want to detect
    colors_to_detect = ['red', 'orange', 'yellow', 'green', 'blue', 
                       'indigo', 'violet', 'white', 'gray', 'black', 'brown']

    cv2.namedWindow("Person Segmentation with Color Detection")
    
    # For showing the output of the brightness adjust
    cv2.namedWindow("Brightness Adjusted Preview")
    cv2.createTrackbar("Brightness", "Brightness Adjusted Preview", 100, 200, nothing)


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror view
        # frame = cv2.flip(frame, 1)
        
        # Run YOLOv8 segmentation
        results = model(frame, classes=[0])  # 0 is the class ID for person
        
        # Initialize person mask
        person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Process segmentation results
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    # Convert mask to numpy array
                    mask = mask.cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    
                    # Resize mask to match frame dimensions
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    
                    # Threshold to create binary mask
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    
                    # Add to person mask
                    person_mask = cv2.bitwise_or(person_mask, mask)
        
        # Perform color detection within person mask
        if np.max(person_mask) > 0:
            output_frame, result, objects = detect_colors_in_segmented_person(
                frame, colors_to_detect, person_mask)
            # Apply the mask to the frame to isolate the person region
            person_region = cv2.bitwise_and(frame, frame, mask=person_mask)
            person_brightness = calculate_brightness(person_region)


            # Checking the person if its too dark or not.
            if person_brightness < 10:
                print("Person too dark, skipping detection.")
                continue

        else:
            output_frame = frame.copy()
            result = np.zeros_like(frame)
            objects = []
        
        # Display information
        cv2.putText(output_frame, f"Detecting: {', '.join(colors_to_detect)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, f"Color regions found: {len(objects)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_frame, "Press Q to quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # For showing the output of the whole frame's brightness uncomment the code below to test.
        # brightness = calculate_brightness(frame)
        # cv2.putText(output_frame, f"Brightness: {brightness:.1f}", (10, 120),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # For showing the output of the brightness of the person.
        cv2.putText(output_frame, f"Person Brightness: {person_brightness:.1f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show results
        cv2.imshow('Person Segmentation with Color Detection', output_frame)
        cv2.imshow('Segmented Colors', result)

        # Get brightness value from trackbar (-100 to +100)
        brightness_value = cv2.getTrackbarPos('Brightness', 'Brightness Adjusted Preview') - 100
      
        # Apply brightness adjustment only to the person region
        bright_person_region = adjust_brightness(person_region, brightness_value)
        brighter_person = calculate_brightness(bright_person_region)
        print(brighter_person)

        cv2.imshow("Brightness Adjusted Preview", bright_person_region)

        # Merge the brightened person region back into the frame
        frame_adjusted = frame.copy()
        frame_adjusted[person_mask > 0] = bright_person_region[person_mask > 0]

        output_frame, result, objects = detect_colors_in_segmented_person(
    frame_adjusted, colors_to_detect, person_mask)
        
    # For showing the output of the brightness of the person.
        cv2.putText(output_frame, f"Person Brightness: {brighter_person:.1f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()