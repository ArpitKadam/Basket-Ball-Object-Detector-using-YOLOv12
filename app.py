import os
import cv2
import subprocess
import platform
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from cvzone import putTextRect
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
MODEL_FOLDER = 'Models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Platform-specific ffmpeg path
def get_ffmpeg_path():
    if platform.system() == "Windows":
        return os.path.join("ffmpeg", "bin", "ffmpeg.exe")
    else:
        return "ffmpeg"

# Enhanced bounding box drawing with better styling
def draw_boxes(frame, results, model):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f'{model.names[cls_id]} {conf:.2f}'
            
            # Enhanced styling with gradient-like effect
            color = (147, 20, 255)  # Purple color matching the theme
            thickness = 3
            
            # Draw main rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw corner accents for modern look
            corner_length = 20
            corner_thickness = 4
            
            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
            
            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
            
            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
            
            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
            
            # Enhanced label with better styling
            putTextRect(frame, label, (x1, y1 - 15), scale=1.2, thickness=2, 
                       colorR=color, colorT=(255, 255, 255), offset=8)
    
    return frame

# Convert to H.264 using ffmpeg
def convert_to_h264(input_path, output_path):
    ffmpeg_path = get_ffmpeg_path()
    temp_path = output_path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run([
            ffmpeg_path, "-y",
            "-i", input_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-movflags", "+faststart",
            "-preset", "fast",  # Faster encoding
            "-crf", "23",       # Better quality
            temp_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_path, output_path)
    except FileNotFoundError:
        print("⚠️ FFmpeg not found. Ensure it's bundled or installed and in PATH.")
    except subprocess.CalledProcessError:
        print("⚠️ FFmpeg failed during conversion.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        model_choice = request.form['model']
        
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            model_path = os.path.join(MODEL_FOLDER, model_choice)
            model = YOLO(model_path)
            
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                output_filename = os.path.splitext(filename)[0] + '_detected.jpg'
            else:
                output_filename = os.path.splitext(filename)[0] + '_detected.mp4'
            
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            file.save(input_path)
            
            # Process video with enhanced quality
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                cap = cv2.VideoCapture(input_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    results = model(frame, verbose=False, conf=0.5)  # Higher confidence threshold
                    frame = draw_boxes(frame, results, model)
                    out.write(frame)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:  # Print progress every 30 frames
                        progress = (frame_count / total_frames) * 100
                        print(f"Processing: {progress:.1f}% complete")
                
                cap.release()
                out.release()
                
                # Convert to H.264 for better web compatibility
                convert_to_h264(output_path, output_path)
            
            # Process image with enhanced quality
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = cv2.imread(input_path)
                results = model(image, verbose=False, conf=0.5)  # Higher confidence threshold
                image = draw_boxes(image, results, model)
                
                # Save with higher quality
                cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            input_url = input_path.replace('\\', '/')
            output_url = output_path.replace('\\', '/')
            
            return render_template('home.html', output_file=output_url, input_file=input_url)
    
    return render_template('home.html', output_file=None)

@app.route('/static/output/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
