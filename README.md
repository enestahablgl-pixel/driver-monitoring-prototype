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

## Entry point

Run `python app.py` from the project directory after setting up
the project's dependencies.

- `n` — toggle facial landmark overlays.
- `q` — quit the application.
