Who got the bigger slice!
Basic Details
Team Name: Thoplikanavila makale
Team Members
Team Lead: Jual Aju - Sahrdaya College of Engineering and Technology
Member 2: Angel Rose K T - Sahrdaya College of Engineering and Technology

Project Description
"Who Got the Bigger Slice" is a lighthearted web application that uses image analysis to settle the age-old argument of unequal portions. Simply upload a photo of pizza, cake, or any shared food, and the app humorously calculates who got the bigger slice. Combining computer vision with playful commentary, it transforms food-sharing into a fun, slightly competitive experience—perfect for friends, family, or anyone who suspects portion injustice at the dinner table.

The Problem (that doesn't exist)
Sometimes, you’re eating with friends, and someone swears your slice is bigger. Or maybe you think they got more fries. It’s not a real problem… but it still starts debates, side-eyes, and lifelong grudges over a few extra crumbs. We’re here to settle these “portion wars” once and for all—because life’s too short to keep wondering who got the bigger bite.


The Solution (that nobody asked for)
We built a website that uses OpenCV magic to judge your food photos and declare who scored the bigger portion. Pizza, cake, fries — nothing escapes its pixel-level judgement. It’s totally unnecessary, slightly petty, and 100% perfect for settling those “No, I got less!” arguments. Basically, it’s technology used for the truly important things in life… like proving your sibling’s a greedy little goblin.

Technical Details
Technologies/Components Used
For Software:cv2,yolo,tkinter

Languages used : 
python

Frameworks used :
Ultralytics YOLOv8

Libraries used :
opencv
numpy

Tools used :
Python 3 – The main programming language.

Ultralytics YOLOv8 – Object detection framework to spot the food slices.

OpenCV – For webcam access, image processing, and contour drawing.

NumPy – For area calculation and numerical operations.

Webcam – Your super scientific food-measuring device.

Terminal / Command Prompt – To run the magic.



Installation

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



Project Documentation
For Software:

Screenshots (Add at least 3)
WhatsApp Image 2025-08-09 at 06.20.56_0fafb2f3.jpg

![Screenshot2](Add screenshot 2 here with proper name) Add caption explaining what this shows

![Screenshot3](Add screenshot 3 here with proper name) Add caption explaining what this shows

Diagrams
![Workflow](Add your workflow/architecture diagram here) Add caption explaining your workflow

For Hardware:

Schematic & Circuit
![Circuit](Add your circuit diagram here) Add caption explaining connections

![Schematic](Add your schematic diagram here) Add caption explaining the schematic

Build Photos
![Components](Add photo of your components here) List out all components shown

![Build](Add photos of build process here) Explain the build steps

![Final](Add photo of final product here) Explain the final build

Project Demo
Video
[Add your demo video link here] Explain what the video demonstrates

Additional Demos
[Add any extra demo materials/links]

Team Contributions
[Name 1]: [Specific contributions]
[Name 2]: [Specific contributions]
[Name 3]: [Specific contributions]
Made with ❤️ at TinkerHub Useless Projects
