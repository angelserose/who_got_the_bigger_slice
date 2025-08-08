import cv2
import numpy as np
from ultralytics import YOLO
import torch
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

print("Loading YOLOv8m model... This may take a moment...")
# Initialize YOLO with medium model for balanced speed and accuracy
model = YOLO('yolov8m.pt')  # Using YOLOv8m instead of YOLOv8x
model.conf = 0.3    # Slightly higher confidence threshold for more precise detection
model.iou = 0.45    # IOU threshold for good multiple object detection
model.max_det = 5   # Maximum number of detections per image

# Warm up the model
warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
model(warmup_frame)
print("Model loaded successfully!")

def enhance_image(frame):
    # Enhance image quality
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Moderate denoising (faster than with x model)
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 8, 8, 7, 15)
    return frame

def get_food_pieces(frame):
    # Enhance image
    frame = enhance_image(frame)
    
    # Run detection with YOLOv8m
    results = model(frame, stream=True)
    result = next(results)
    
    # Food-related classes in COCO dataset
    food_classes = [
        'pizza', 'sandwich', 'hot dog', 'carrot', 'cake', 'donut', 'cookie',
        'apple', 'orange', 'banana', 'broccoli', 'bread', 'cake', 'pizza slice',
        'bowl', 'plate', 'dining table', 'food'
    ]
    
    pieces = []
    detected_areas = []
    
    # Process YOLO detections
    for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
        class_name = model.names[int(cls)]
        if class_name in food_classes and conf > 0.3:  # Increased confidence threshold
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            
            # Create mask for the detected region
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Convert ROI to HSV for better segmentation
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Create mask using adaptive thresholding
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                binary_roi = cv2.adaptiveThreshold(
                    gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 21, 10
                )
                
                # Clean up the mask
                kernel = np.ones((5,5), np.uint8)
                binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel)
                binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel)
                
                # Place the processed mask in the original image size
                mask[y1:y2, x1:x2] = binary_roi
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 1000:  # Minimum area threshold
                        # Get convex hull for better accuracy
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        
                        # Convert hull to list of points for JSON
                        points = hull.reshape(-1, 2).tolist()
                        
                        # Store detection info
                        detected_areas.append({
                            'area': hull_area,
                            'contour': points,
                            'confidence': float(conf),
                            'class': class_name
                        })
    
    # Sort detected areas by size
    detected_areas.sort(key=lambda x: x['area'], reverse=True)
    
    # Convert to final format
    for i, area_info in enumerate(detected_areas, 1):
        pieces.append({
            'id': i,
            'area': area_info['area'],
            'contour': area_info['contour'],
            'size_ratio': area_info['area'] / detected_areas[0]['area'] if i > 1 else 1.0,
            'confidence': area_info['confidence'],
            'class': area_info['class']
        })
    
    return pieces

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
        
        # Get food pieces
        pieces = get_food_pieces(frame)
        
        # Debug output
        print(f"Found {len(pieces)} pieces")
        for p in pieces:
            print(f"Piece {p['id']}: Area = {p['area']}, Class = {p['class']}, Confidence = {p['confidence']}")
        
        return jsonify({
            'success': True,
            'pieces': pieces
        })
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
