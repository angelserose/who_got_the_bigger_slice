import cv2
import numpy as np
from ultralytics import YOLO
import torch
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

print("Loading YOLOv8x model... This may take a moment...")
# Initialize YOLO with the most accurate model
model = YOLO('yolov8x.pt')  # Using YOLOv8x instead of YOLOv8n
model.conf = 0.25  # Lower confidence threshold for better detection
model.iou = 0.3    # Lower IOU threshold for better detection of close objects
model.max_det = 5  # Maximum number of detections per image

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
    
    # Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    return frame

def get_food_pieces(frame):
    # Enhance image
    frame = enhance_image(frame)
    
    # Run detection with YOLOv8x at multiple scales
    scales = [1.0, 0.8, 1.2]  # Try different scales
    all_detections = []
    
    for scale in scales:
        # Resize image
        height, width = frame.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Run YOLO detection
        results = model(resized, stream=True)
        result = next(results)
        
        # Scale back the coordinates
        scale_x = width / new_width
        scale_y = height / new_height
        
        # Food-related classes in COCO dataset
        food_classes = [
            'pizza', 'sandwich', 'hot dog', 'carrot', 'cake', 'donut', 'cookie',
            'apple', 'orange', 'banana', 'broccoli', 'bread', 'cake', 'pizza slice',
            'bowl', 'plate', 'dining table', 'food'
        ]
        
        # Process detections
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            class_name = model.names[int(cls)]
            if class_name in food_classes and conf > 0.25:
                x1, y1, x2, y2 = box.cpu().numpy()
                # Scale coordinates back
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Create and process mask
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Multi-channel processing
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    
                    # Adaptive thresholding
                    binary_roi = cv2.adaptiveThreshold(
                        gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV, 21, 10
                    )
                    
                    # Clean up mask
                    kernel = np.ones((5,5), np.uint8)
                    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel)
                    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel)
                    
                    mask[y1:y2, x1:x2] = binary_roi
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 1000:  # Minimum area threshold
                            hull = cv2.convexHull(cnt)
                            hull_area = cv2.contourArea(hull)
                            points = hull.reshape(-1, 2).tolist()
                            
                            all_detections.append({
                                'area': hull_area,
                                'contour': points,
                                'confidence': float(conf),
                                'class': class_name
                            })
    
    # Remove overlapping detections
    filtered_detections = []
    all_detections.sort(key=lambda x: x['area'] * x['confidence'], reverse=True)
    
    for det in all_detections:
        # Check if this detection overlaps significantly with any existing one
        is_unique = True
        for existing in filtered_detections:
            # Simple overlap check using bounding boxes of contours
            if len(set(map(tuple, det['contour'])) & set(map(tuple, existing['contour']))) > 0:
                is_unique = False
                break
        if is_unique:
            filtered_detections.append(det)
    
    # Convert to final format
    pieces = []
    for i, det in enumerate(filtered_detections, 1):
        pieces.append({
            'id': i,
            'area': det['area'],
            'contour': det['contour'],
            'size_ratio': det['area'] / filtered_detections[0]['area'] if i > 1 else 1.0,
            'confidence': det['confidence'],
            'class': det['class']
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
