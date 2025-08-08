# Who Got The Bigger Slice

A Python application that uses YOLOv8 and OpenCV to detect and compare food pieces.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/angelserose/who_got_the_bigger_slice.git
cd who_got_the_bigger_slice
```

2. Download the required YOLO models:
```bash
# Download YOLOv8m model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# For other models (optional):
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
# wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

3. Install the required packages:
```bash
pip install ultralytics opencv-python numpy
```

## Usage

Run the main script:
```bash
python main.py
```

The application will:
1. Open your webcam
2. Detect food items in real-time
3. Draw bounding boxes and contours
4. Show area measurements for comparison

Press 'q' to quit the application.

## Note

The YOLO model files (*.pt) are not included in this repository due to their size. Please download them separately using the commands above.
