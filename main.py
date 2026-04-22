import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import deque
import time
import serial
import serial.tools.list_ports

# Configuration
MODEL_PATH = "yolov8n.pt"
LOW_STOCK_THRESH = 2
MID_STOCK_THRESH = 5
MAX_HISTORY = 500
INFER_EVERY_N = 5

GROCERY_MAP = {
    "bottle": "Drinks",
    "banana": "Fruits",
    "apple": "Fruits",
    "orange": "Fruits",
    "cup": "Utensils",
    "sandwich": "Food",
    "bowl": "Food",
    "fork": "Utensils",
    "knife": "Utensils",
    "spoon": "Utensils",
    "wine glass": "Drinks",
    "carrot": "Vegetables",
    "broccoli": "Vegetables",
    "pizza": "Food",
    "cake": "Food",
    "donut": "Food",
}

# Page configuration
st.set_page_config(
    page_title="Smart Inventory",
    layout="wide"
)

# ESP32 Functions
def get_available_ports():
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports] if ports else ["No ports found"]

def connect_esp32(port, baud=115200):
    try:
        s = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        return s
    except:
        return None

def send_to_esp32(ser, total_count):
    if ser and ser.is_open:
        try:
            ser.write(f"{total_count}\n".encode())
        except:
            pass

# Sidebar
with st.sidebar:
    st.title("Smart Inventory")

    camera_index = st.selectbox("Camera", [0, 1, 2, 3])
    run = st.button("Start")
    stop = st.button("Stop")

    available_ports = get_available_ports()
    selected_port = st.selectbox("Port", available_ports)

    baud_rate = st.selectbox("Baud", [9600, 115200], index=1)
    connect_btn = st.button("Connect ESP32")
    disconnect_btn = st.button("Disconnect ESP32")

    low_t = st.slider("Low Stock Threshold", 1, 5, LOW_STOCK_THRESH)
    mid_t = st.slider("Mid Stock Threshold", 2, 10, MID_STOCK_THRESH)
    skip_n = st.slider("Frame Skip", 1, 6, INFER_EVERY_N)
    conf = st.slider("Confidence", 0.1, 0.9, 0.4)

# Load model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

# Session state
if "cap" not in st.session_state:
    st.session_state.cap = None
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=MAX_HISTORY)
if "esp32" not in st.session_state:
    st.session_state.esp32 = None
if "led_state" not in st.session_state:
    st.session_state.led_state = False

# ESP32 connection
if connect_btn:
    st.session_state.esp32 = connect_esp32(selected_port, baud_rate)

if disconnect_btn and st.session_state.esp32:
    st.session_state.esp32.close()
    st.session_state.esp32 = None

# Camera control
if run and st.session_state.cap is None:
    st.session_state.cap = cv2.VideoCapture(camera_index)

if stop and st.session_state.cap:
    st.session_state.cap.release()
    st.session_state.cap = None

cap = st.session_state.cap

st.title("Smart Grocery Inventory System")

if cap is None:
    st.info("Camera not started")
    st.stop()

# Main loop
ret, frame = cap.read()
if not ret:
    st.error("Camera error")
    st.stop()

frame = cv2.resize(frame, (640, 480))
results = model(frame, conf=conf)

inventory = {}
for r in results:
    for box in r.boxes:
        label = model.names[int(box.cls[0])]
        inventory[label] = inventory.get(label, 0) + 1

# Display frame
annotated = results[0].plot()
st.image(annotated, channels="BGR")

# Data processing
df = pd.DataFrame(list(inventory.items()), columns=["Item", "Count"])

total = int(df["Count"].sum()) if not df.empty else 0
unique = len(df)
low = int((df["Count"] < low_t).sum()) if not df.empty else 0

st.metric("Total Items", total)
st.metric("Unique Items", unique)
st.metric("Low Stock Items", low)

# LED Control
led_on = total < low_t
if led_on != st.session_state.led_state:
    send_to_esp32(st.session_state.esp32, total)
    st.session_state.led_state = led_on

st.write("LED Status:", "ON" if led_on else "OFF")

# Charts
if not df.empty:
    fig = px.bar(df, x="Item", y="Count")
    st.plotly_chart(fig)

# Table
st.dataframe(df)