# Smart Inventory Management System

## Micro Project-II (Semester IV - IIoT)

### Team Members:
- Ramanpreet kaur Renu  (Roll No: 24009064)
- R Bhavya Naidu (Roll No: 24009051)
- Abhaya Deogade (Roll No: 24009058)


## Project Description
This project is an AI-based Smart Inventory Management System that detects and counts grocery items using computer vision. It uses YOLOv8 for object detection and Streamlit for dashboard visualization.

The system also integrates with ESP32 to indicate low stock using an LED.

## Technologies Used
- Python
- OpenCV
- YOLOv8
- Streamlit
- ESP32
## Components Used
- Webcam
- ESP32
- LED
- Computer
## Working
1. Camera captures live video.
2. YOLO detects objects.
3. Items are counted.
4. Data is shown on dashboard.
5. LED turns ON when stock is low.
## How to Run
1. Install libraries:
   pip install streamlit opencv-python ultralytics pandas plotly pyserial

2. Run:
   streamlit run main.py
## Output
- Live detection
- Inventory tracking
- Low stock alert (LED)
