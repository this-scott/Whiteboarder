import cv2
import numpy as np

def detect_rectangles(frame):
    """
    Detect rectangles in the frame and return their contours.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has 4 vertices (rectangle)
        if len(approx) == 4:
            # Calculate area to filter out small detections
            area = cv2.contourArea(approx)
            if area > 1000:  # Minimum area threshold
                rectangles.append(approx)
    
    return rectangles

def main():
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    print("Press '+' to increase edge detection sensitivity")
    print("Press '-' to decrease edge detection sensitivity")
    
    # Edge detection thresholds (can be adjusted)
    canny_low = 50
    canny_high = 150
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Create a copy for drawing
        output = frame.copy()
        
        # Detect rectangles
        rectangles = detect_rectangles(frame)
        
        # Draw rectangles on the output frame
        for rect in rectangles:
            # Draw the rectangle outline in green
            cv2.drawContours(output, [rect], -1, (0, 255, 0), 3)
            
            # Draw circles at the corners in red
            for point in rect:
                cv2.circle(output, tuple(point[0]), 5, (0, 0, 255), -1)
        
        # Display detection count
        cv2.putText(output, f"Rectangles detected: {len(rectangles)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('Rectangle Detection', output)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('+'):
            canny_low = min(canny_low + 10, 200)
            canny_high = min(canny_high + 10, 300)
            print(f"Sensitivity increased: {canny_low}/{canny_high}")
        elif key == ord('-'):
            canny_low = max(canny_low - 10, 10)
            canny_high = max(canny_high - 10, 50)
            print(f"Sensitivity decreased: {canny_low}/{canny_high}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()