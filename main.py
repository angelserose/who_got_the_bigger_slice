import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import os

class CookieDetectionSystem:
    def __init__(self, model_path='yolov8m.pt', conf_threshold=0.25):
        """Initialize the cookie detection system with GUI."""
        print("Loading YOLOv8 model... This may take a moment...")
        self.model = YOLO(model_path)
        self.model.conf = conf_threshold
        self.model.iou = 0.45
        self.model.max_det = 15
        
        # Cookie and biscuit related classes (expanded)
        self.cookie_classes = {
            'cookie', 'donut', 'cake', 'sandwich cookie', 'biscuit'
        }
        
        # All food classes for general detection
        self.food_classes = {
            'pizza', 'sandwich', 'hot dog', 'carrot', 'cake', 'donut', 'cookie',
            'apple', 'orange', 'banana', 'broccoli', 'bread', 'bowl', 'plate',
            'dining table', 'spoon', 'fork', 'knife', 'cup', 'bottle'
        }
        
        # Camera and GUI variables
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detection_results = []
        
        # Performance tracking
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # Create GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Create the GUI interface."""
        self.root = tk.Tk()
        self.root.title("Cookie & Biscuit Detection System")
        self.root.geometry("1000x700")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding="5")
        self.video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Video label
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack()
        
        # Control buttons frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        
        # Buttons
        self.start_button = ttk.Button(controls_frame, text="Start Camera", command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=5, pady=2)
        
        self.stop_button = ttk.Button(controls_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5, pady=2)
        
        self.photo_button = ttk.Button(controls_frame, text="📸 Take Photo & Scan for Cookies", 
                                      command=self.take_photo_and_scan, state="disabled")
        self.photo_button.grid(row=0, column=2, padx=5, pady=2)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="5")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Results text area with scrollbar
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(text_frame, height=8, width=40, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Click 'Start Camera' to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Cookie count display
        self.cookie_count_var = tk.StringVar()
        self.cookie_count_var.set("Cookies Detected: 0")
        cookie_count_label = ttk.Label(main_frame, textvariable=self.cookie_count_var, 
                                      font=("Arial", 12, "bold"), foreground="blue")
        cookie_count_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def enhance_image(self, frame):
        """Enhanced image preprocessing for better cookie detection."""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply bilateral filter for cookie texture preservation
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        return frame
    
    def detect_cookies_advanced(self, roi):
        """Advanced cookie detection using multiple techniques."""
        if roi.size == 0:
            return None, []
            
        # Convert to different color spaces
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Cookie-specific color ranges (brown, golden, etc.)
        cookie_colors = [
            ([10, 50, 20], [25, 255, 200]),    # Brown cookies
            ([15, 30, 100], [35, 255, 255]),   # Golden cookies
            ([0, 0, 150], [180, 30, 255])      # Light colored cookies
        ]
        
        combined_mask = np.zeros_like(gray)
        
        for lower, upper in cookie_colors:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Additional edge-based detection for cookie boundaries
        edges = cv2.Canny(gray, 50, 150)
        combined_mask = cv2.bitwise_or(combined_mask, edges)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # Find circular/elliptical shapes (typical cookie shapes)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cookie_pieces = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 800:  # Minimum cookie size
                # Check if shape is somewhat circular (cookies are usually round)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Reasonable circularity for cookies
                        cookie_pieces.append(cnt)
        
        return combined_mask, cookie_pieces
    
    def process_frame_for_cookies(self, frame):
        """Process frame specifically looking for cookies and biscuits."""
        enhanced_frame = self.enhance_image(frame.copy())
        
        # Run YOLO detection
        results = self.model(enhanced_frame, stream=True, verbose=False)
        result = next(results)
        
        detected_items = []
        cookie_count = 0
        output_frame = frame.copy()
        
        if result.boxes is not None:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                class_name = self.model.names[int(cls)]
                
                # Check for any food item first, then prioritize cookies
                if class_name in self.food_classes and conf > self.model.conf:
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    
                    # Special handling for cookies and biscuits
                    is_cookie = class_name in self.cookie_classes
                    color = (0, 255, 0) if is_cookie else (255, 0, 0)  # Green for cookies, red for other food
                    
                    # Draw bounding box
                    thickness = 3 if is_cookie else 2
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Label with special marking for cookies
                    prefix = "🍪 " if is_cookie else ""
                    label = f'{prefix}{class_name}: {conf:.2f}'
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(output_frame, (x1, y1 - label_size[1] - 15), 
                                (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(output_frame, label, (x1 + 5, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Advanced cookie analysis
                    if is_cookie:
                        roi = enhanced_frame[y1:y2, x1:x2]
                        mask, cookie_pieces = self.detect_cookies_advanced(roi)
                        
                        if cookie_pieces:
                            total_cookie_area = 0
                            for piece in cookie_pieces:
                                area = cv2.contourArea(piece)
                                total_cookie_area += area
                                
                                # Draw cookie pieces
                                piece_shifted = piece + np.array([x1, y1])
                                cv2.drawContours(output_frame, [piece_shifted], -1, (0, 255, 255), 2)
                            
                            cookie_count += len(cookie_pieces)
                            
                            detected_items.append({
                                'class': class_name,
                                'confidence': float(conf),
                                'is_cookie': True,
                                'piece_count': len(cookie_pieces),
                                'total_area': float(total_cookie_area),
                                'bbox': (x1, y1, x2, y2)
                            })
                    else:
                        detected_items.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'is_cookie': False,
                            'bbox': (x1, y1, x2, y2)
                        })
        
        return output_frame, detected_items, cookie_count
    
    def update_video_feed(self):
        """Update the video feed in the GUI."""
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                
                # Process frame
                processed_frame, detections, cookie_count = self.process_frame_for_cookies(frame)
                
                # Update cookie count
                self.cookie_count_var.set(f"Cookies Detected: {cookie_count}")
                
                # Convert to RGB and resize for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (640, 480))
                
                # Convert to PhotoImage
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update video label
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
                # Schedule next update
                self.root.after(30, self.update_video_feed)
    
    def start_camera(self):
        """Start the camera feed."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.photo_button.configure(state="normal")
        self.status_var.set("Camera running - Ready to detect cookies!")
        
        # Start video feed
        self.update_video_feed()
    
    def stop_camera(self):
        """Stop the camera feed."""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.photo_button.configure(state="disabled")
        self.status_var.set("Camera stopped")
        
        # Clear video display
        self.video_label.configure(image="")
        self.video_label.image = None
    
    def take_photo_and_scan(self):
        """Take a photo and scan it specifically for cookies and biscuits."""
        if not self.current_frame is not None:
            messagebox.showwarning("Warning", "No frame available! Make sure camera is running.")
            return
        
        self.status_var.set("Taking photo and scanning for cookies...")
        
        # Save the photo
        timestamp = int(time.time())
        photo_filename = f"cookie_scan_{timestamp}.jpg"
        cv2.imwrite(photo_filename, self.current_frame)
        
        # Process the photo for detailed cookie analysis
        processed_frame, detections, cookie_count = self.process_frame_for_cookies(self.current_frame)
        
        # Save processed image
        processed_filename = f"cookie_analysis_{timestamp}.jpg"
        cv2.imwrite(processed_filename, processed_frame)
        
        # Update results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"📸 Photo Analysis Results (Timestamp: {timestamp})\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")
        
        cookie_items = [item for item in detections if item['is_cookie']]
        other_items = [item for item in detections if not item['is_cookie']]
        
        if cookie_items:
            self.results_text.insert(tk.END, f"🍪 COOKIES FOUND: {len(cookie_items)} types\n")
            self.results_text.insert(tk.END, f"Total Cookie Pieces: {cookie_count}\n\n")
            
            for i, item in enumerate(cookie_items, 1):
                self.results_text.insert(tk.END, f"{i}. {item['class'].upper()}\n")
                self.results_text.insert(tk.END, f"   Confidence: {item['confidence']:.2f}\n")
                if 'piece_count' in item:
                    self.results_text.insert(tk.END, f"   Individual pieces: {item['piece_count']}\n")
                    self.results_text.insert(tk.END, f"   Total area: {item['total_area']:.0f} pixels²\n")
                self.results_text.insert(tk.END, "\n")
        else:
            self.results_text.insert(tk.END, "❌ No cookies or biscuits detected in this photo.\n\n")
        
        if other_items:
            self.results_text.insert(tk.END, f"🍽️ Other food items found: {len(other_items)}\n")
            for item in other_items:
                self.results_text.insert(tk.END, f"   - {item['class']} (confidence: {item['confidence']:.2f})\n")
        
        self.results_text.insert(tk.END, f"\n📁 Photos saved:\n")
        self.results_text.insert(tk.END, f"   Original: {photo_filename}\n")
        self.results_text.insert(tk.END, f"   Analysis: {processed_filename}\n")
        
        self.status_var.set(f"Scan complete! Found {cookie_count} cookie pieces in {len(cookie_items)} cookie types.")
        
        # Show completion message
        message = f"Scan complete!\n\nFound {cookie_count} cookie pieces"
        if cookie_items:
            message += f" across {len(cookie_items)} different cookie types."
        else:
            message += ".\nNo cookies detected in this image."
        
        messagebox.showinfo("Cookie Scan Results", message)
    
    def run(self):
        """Run the application."""
        # Warm up model
        print("Warming up the model...")
        warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(warmup_frame, verbose=False)
        print("Model ready!")
        
        # Handle window close
        def on_closing():
            self.stop_camera()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

def main():
    """Main function to run the cookie detection system."""
    app = CookieDetectionSystem(conf_threshold=0.2)  # Lower threshold for better cookie detection
    app.run()

if __name__ == "__main__":
    main()