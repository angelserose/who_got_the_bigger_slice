import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

# Initialize YOLO model with better parameters
model = YOLO('yolov8n.pt')
model.conf = 0.3  # Lower confidence threshold for better detection
model.iou = 0.4   # Lower IOU threshold for better multiple object detection

def adjust_exposure(frame):
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge channels
    adjusted = cv2.merge((cl,a,b))
    
    # Convert back to BGR
    return cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)

def enhance_image(frame):
    # Adjust exposure
    frame = adjust_exposure(frame)
    
    # Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    # Enhance contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    adjusted = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(adjusted, cv2.COLOR_LAB2BGR)
    
    return frame

def get_food_masks(frame):
    # Enhance image first
    enhanced_frame = enhance_image(frame)
    
    # Run YOLO detection with confidence
    results = model(enhanced_frame, stream=True)
    result = next(results)
    
    # Food classes with common variations
    food_classes = [
        'pizza', 'sandwich', 'hot dog', 'carrot', 'cake', 'donut', 'cookie',
        'apple', 'orange', 'banana', 'broccoli', 'bread', 'burger', 'pizza slice',
        'fruit', 'vegetable', 'pastry', 'food'
    ]
    
    masks = []
    detected_objects = []
    
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        class_name = model.names[int(cls)]
        if class_name in food_classes and conf > 0.3:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            
            # Create detailed mask using additional processing
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            roi = enhanced_frame[y1:y2, x1:x2]
            
            if roi.size > 0:  # Check if ROI is valid
                # Convert ROI to HSV for better color segmentation
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Create mask using color thresholding
                food_mask = cv2.inRange(hsv_roi, np.array([0,20,20]), np.array([180,255,255]))
                
                # Apply morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel)
                food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel)
                
                # Place the processed mask in the original image size
                mask[y1:y2, x1:x2] = food_mask
                
                masks.append(mask)
                detected_objects.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': [x1, y1, x2, y2]
                })
    
    return masks, detected_objects

def get_contour_areas(frame):
    # Get food masks and detection info from YOLO
    food_masks, detected_objects = get_food_masks(frame)
    
    if not food_masks:
        # Enhanced fallback method
        enhanced_frame = enhance_image(frame)
        
        # Multi-channel processing
        hsv = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Create masks using different methods
        # HSV color-based mask
        food_mask_hsv = cv2.inRange(hsv, np.array([0,20,20]), np.array([180,255,255]))
        
        # Adaptive threshold mask
        thresh_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Combine masks
        combined_mask = cv2.bitwise_or(food_mask_hsv, thresh_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        food_masks = [combined_mask]
    
    contour_info = []
    piece_num = 1
    
    for mask in food_masks:
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process each contour
        min_area = 1000
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Smooth the contour
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Calculate actual area using convex hull for better accuracy
                hull = cv2.convexHull(approx)
                hull_area = cv2.contourArea(hull)
                
                contour_info.append((hull_area, hull, piece_num))
                piece_num += 1
    
    return sorted(contour_info, key=lambda x: x[0], reverse=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get image data from request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to cv2 image
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process image
        contour_info = get_contour_areas(frame)
        
        # Prepare results
        pieces = []
        for area, cnt, piece_num in contour_info:
            # Convert contour to list of points for JSON
            contour_points = cnt.reshape(-1, 2).tolist()
            
            pieces.append({
                'id': piece_num,
                'area': float(area),
                'contour': contour_points,
                'size_ratio': float(area) / float(contour_info[-1][0]) if len(contour_info) > 1 else 1.0
            })
        
        return jsonify({
            'success': True,
            'pieces': pieces
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def main():
    # Initialize webcam with improved settings
    webcam_index = 0
    cap = None
    
    while webcam_index < 2:
        cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
        if cap is not None and cap.isOpened():
            # Configure camera settings for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
            
            ret, test_frame = cap.read()
            if ret:
                break
        if cap is not None:
            cap.release()
        webcam_index += 1
    
    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam.")
        print("Please ensure:")
        print("1. Your webcam is properly connected")
        print("2. No other application is using the webcam")
        print("3. Your webcam drivers are installed correctly")
        return
    
    print("Loading YOLO model... please wait...")
    # Warm up the YOLO model
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy_frame)
    
    print("Ready!")
    print("Press 'c' to capture and compare pieces")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Show the original frame
        cv2.imshow('Webcam Feed', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Get contours and their areas
            contour_info = get_contour_areas(frame)
            
            # Draw contours and show results
            result_frame = frame.copy()
            
            if len(contour_info) < 2:
                cv2.putText(result_frame, "Please show at least 2 pieces", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Draw contours and label them
                for area, cnt, piece_num in contour_info:
                    # Draw contour
                    cv2.drawContours(result_frame, [cnt], -1, (0, 255, 0), 2)
                    
                    # Calculate centroid for text placement
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    
                    # Add piece number and area
                    cv2.putText(result_frame, f"Piece {piece_num}", 
                              (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"Area: {int(area)}", 
                              (cx - 40, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show which piece is biggest
                biggest_piece = contour_info[0]
                cv2.putText(result_frame, f"Piece {biggest_piece[2]} is the biggest!", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Result', result_frame)
        
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
