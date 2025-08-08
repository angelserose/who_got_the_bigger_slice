import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

def preprocess_image(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def get_pieces(frame):
    # Preprocess the image
    processed = preprocess_image(frame)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and process contours
    pieces = []
    min_area = 1000  # Minimum area threshold
    
    for i, cnt in enumerate(contours, 1):
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Get convex hull for more accurate area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            
            # Convert hull points to list for JSON
            points = hull.reshape(-1, 2).tolist()
            
            pieces.append({
                'id': i,
                'area': float(hull_area),
                'contour': points
            })
    
    # Sort pieces by area
    pieces.sort(key=lambda x: x['area'], reverse=True)
    
    # Calculate relative sizes
    if pieces:
        largest_area = pieces[0]['area']
        for piece in pieces:
            piece['size_ratio'] = piece['area'] / largest_area
    
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
        
        # Convert to OpenCV image
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get the pieces
        pieces = get_pieces(frame)
        
        # Debug output
        print(f"Found {len(pieces)} pieces")
        for p in pieces:
            print(f"Piece {p['id']}: Area = {p['area']}")
        
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
