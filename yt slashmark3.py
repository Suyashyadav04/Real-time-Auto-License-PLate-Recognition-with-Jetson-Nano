import cv2
import torch
from yolov5.models.common import DetectMultiBackend

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)
    
    # Draw bounding boxes
    results.render()
    cv2.imshow('License Plate Detection', results.imgs[0])
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def extract_license_plate(frame, results):
    for result in results.xyxy[0]:  # Loop through detected objects
        # Check if it's a license plate class (class ID can vary based on your model)
        if int(result[-1]) == LICENSE_PLATE_CLASS_ID:
            x1, y1, x2, y2 = map(int, result[:4])  # Extract bounding box coordinates
            plate = frame[y1:y2, x1:x2]
            return plate
    return None
import pytesseract

def recognize_text(plate_image):
    # Preprocess image for OCR
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Perform OCR
    text = pytesseract.image_to_string(thresh, config='--psm 8')  # PSM 8: Single word recognition
    return text.strip()
import cv2
import torch
import pytesseract

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    results = model(frame)

    # Extract license plate
    plate_image = extract_license_plate(frame, results)
    if plate_image is not None:
        text = recognize_text(plate_image)
        print(f"License Plate: {text}")
    
    # Display results
    results.render()
    cv2.imshow('License Plate Recognition', results.imgs[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import csv
from datetime import datetime

def save_plate_data(plate_text):
    with open('plates.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([plate_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
