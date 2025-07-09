import os
import cv2
from ultralytics import YOLO
from cvzone import putTextRect
from pathlib import Path

# Define input and output folder paths
INPUT_FOLDER = 'Input'
OUTPUT_FOLDER = 'Output'

model = YOLO('Models\yolov12n.pt')

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper function to draw bounding boxes using cvzone
def draw_boxes(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence
            cls_id = int(box.cls[0])   # Class ID
            label = f'{model.names[cls_id]} {conf:.2f}'
            putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=(255, 0, 255), offset=5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    return frame

# Loop through input folder
for filename in os.listdir(INPUT_FOLDER):
    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    # Check if it's a video file
    if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            success, frame = cap.read()
            if not success:
                break
            results = model(frame, verbose=False)
            frame = draw_boxes(frame, results)
            out.write(frame)

        cap.release()
        out.release()

    # Check if it's an image file
    elif filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image = cv2.imread(input_path)
        results = model(image, verbose=False)
        image = draw_boxes(image, results)
        cv2.imwrite(output_path, image)

print("✅ Inference complete. Check the output folder.")
