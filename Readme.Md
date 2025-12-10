🔥 Fire & Human Detection System
Real-time Fire Detection with Human Presence Alert 

This project is a real-time Fire Detection System that uses a webcam to detect both fire and human presence simultaneously.
It triggers an alarm sound only when fire is detected AND a human is present, reducing false alerts.

It is optimized to detect small flames such as matchsticks and lighters, while maintaining good performance and accuracy.

🚀 Features
🔥 Fire Detection

Detects fire using HSV color segmentation

Tuned to detect very small flames (matchsticks, candles, lighters)

Uses morphological processing to enlarge & stabilize flame detection

Adjustable sensitivity via configuration variables

🧍 Human Detection

Combines:

HOG full-body detector

Haar Cascade face detector

Supports real-time detection on standard webcams

Human detection is throttled for improved FPS

🔔 Alarm System

Plays custom alarm audio (alarm.mp3) using afplay (macOS)

Uses thread-safe logic so multiple alarms do not overlap

Fires the alarm only if:
Fire Detected + Human Present + Stable for N frames

⚡ Performance Optimized

Uses downscaled frames (480×360)

Runs heavy human detection only every N frames

Uses detection caching to keep performance smooth

📁 Project Structure
/Users/omraj/Desktop/Screenshot 2025-12-10 at 2.04.47 PM.png


## Architecture Diagram

/Users/omraj/Desktop/Screenshot 2025-12-10 at 1.56.10 PM.png




🛠️ Technologies Used


Python 3.x

OpenCV (cv2)

NumPy

macOS subsystem audio (afplay)

HOG + SVM person detector

Haar Cascade face detection

📦 Installation
1. Clone the repository
git clone : https://github.com/OmRaj6666/Fire-Detection.git
cd fire-detection-system

2. Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

3. Install required libraries
pip install opencv-python numpy

4. Add your alarm file

Place a sound file named alarm.mp3 inside the project root.

▶️ Running the System

From the project folder:

python3 main.py

Controls

Press q → Quit program

Webcam opens automatically

Two windows:

Fire + Human Detection View

Fire Mask Visualization

🔧 Configuration (Adjust Sensitivity)

Inside main.py, modify these values:

Fire Sensitivity
FIRE_THRESHOLD = 0.0015
MIN_FIRE_PIXELS = 120


Lower values → More sensitive (detects tiny flames)

Higher values → Less sensitive

Human Detection Frequency
HUMAN_DETECT_EVERY = 5


Lower (e.g., 3) → more accurate but slower

Higher (e.g., 8–10) → faster but less frequent detection

🧪 How It Works (Pipeline Overview)
1️⃣ Webcam frame capture

Frames are resized 480×360 for speed.

2️⃣ Human detection

Runs every N frames

Detects:

Full body (HOG)

Face (Haar Cascade)

3️⃣ Fire detection

Converts frame to HSV

Applies fire-color mask

Removes any fire-like pixels inside detected human boxes

Uses dilation to detect small flames

Counts fire pixels & calculates fire ratio

4️⃣ Alarm logic

Alarm rings only when:

fire_detected == True
AND human_detected == True
AND sustained for consecutive frames

🐞 Troubleshooting
❓ Fire not being detected?

Try reducing sensitivity:

FIRE_THRESHOLD = 0.0010
MIN_FIRE_PIXELS = 80

❓ Human not detecting reliably?

Increase face detection priority:

Ensure good lighting

Move closer to webcam

Reduce resize frame size to 640×480 for better HOG accuracy

❓ Alarm not playing?

Check file name: alarm.mp3

Ensure macOS afplay command works:

afplay alarm.mp3

📄 License

This project is open-source. You may modify or use it for personal, academic, or research purposes.

🤝 Contributing

Pull requests are welcome!
If you'd like improvements such as:

YOLO-based human detection

Deep learning fire detection

GPU acceleration

