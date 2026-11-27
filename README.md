# Driver Monitoring Prototype

A computer vision prototype developed during my internship at Visea
İnovatif. It uses camera frames to estimate eye and mouth state
and head orientation, then displays a risk score and visual alerts.

## Features

- Uses facial landmarks to estimate whether the eyes and mouth are
  open or closed.
- Estimates head pitch, yaw, and roll and draws orientation axes
  on the video feed.
- Displays a risk score and an on-screen warning when the alert is active.

## Setup and run on Windows

Use 64-bit Python 3.12 and a webcam. This project uses MediaPipe's legacy
Solutions API. Keep `app.py`, `face_detector.py`, and `landmark_detector.py`
in the same directory.

From the project directory, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "mediapipe==0.10.21" "opencv-contrib-python==4.11.0.86" "numpy==1.26.4"
.\.venv\Scripts\python.exe app.py
```

- `n` — toggle facial landmark overlays.
- `q` — quit the application.
