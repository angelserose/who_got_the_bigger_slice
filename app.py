from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding instead of simple threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return thresh

def get_contour_info(image):
    # Preprocess image
    processed = preprocess_image(image)
    
    # Find contours
    contours, _ = cv2.findContours(
        processed, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and process contours
    min_area = 1000
    pieces = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Simplify contour for web transmission
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Convert contour to list of points
            contour_points = approx.reshape(-1, 2).tolist()
            
            pieces.append({
                'id': i + 1,
                'area': area,
                'contour': contour_points
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
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process image and get results
        pieces = get_contour_info(image)
        
        return jsonify({
            'success': True,
            'pieces': pieces
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
