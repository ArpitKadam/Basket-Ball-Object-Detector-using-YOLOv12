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

# Draw bounding boxes using cvzone
def draw_boxes(frame, results, model):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f'{model.names[cls_id]} {conf:.2f}'
            putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=(255, 0, 255), offset=5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
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
                output_filename = os.path.splitext(filename)[0] + '.jpg'
            else:
                output_filename = os.path.splitext(filename)[0] + '.mp4'

            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            file.save(input_path)

            # Process video
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                cap = cv2.VideoCapture(input_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Compatible on Windows & Docker
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    results = model(frame, verbose=True)
                    frame = draw_boxes(frame, results, model)
                    out.write(frame)

                cap.release()
                out.release()

                # Convert to H.264 for browser
                convert_to_h264(output_path, output_path)

            # Process image
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = cv2.imread(input_path)
                results = model(image, verbose=False)
                image = draw_boxes(image, results, model)
                cv2.imwrite(output_path, image)
            
            input_url = input_path.replace('\\', '/')
            output_url = output_path.replace('\\', '/')
            return render_template('home.html', output_file=output_url, input_file=input_url)

    return render_template('home.html', output_file=None)

@app.route('/static/output/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5000)
