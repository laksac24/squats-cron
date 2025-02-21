import cv2
import numpy as np
import base64
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import poseestimationmodule as pm
from math import floor
from fastapi.middleware.cors import CORSMiddleware
import time
import requests

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = pm.poseDetector()
dir = 0
count = 0
last_change_time = time.time()  # Timer for inactivity check
last_count = 0  # To track count changes

API_URL = "https://pushup-poseestimation-2.onrender.com/process_frame/"  # Your hosted API

class ImageData(BaseModel):
    image: str  # Base64 encoded image

@app.post("/process_frame/")
async def process_frame(data: ImageData):
    global count, dir, last_change_time, last_count

    # Decode base64 image
    image_bytes = base64.b64decode(data.image)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame = detector.findPose(frame, draw=False)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        angle = detector.findAngle(frame, 24, 26, 28, draw=True)
        angle2 = detector.findAngle(frame, 12, 24, 26, draw=True)
        detector.findAngle(frame, 24, 36, 28, draw=True)

        if angle2:
            if 75 < angle2 < 180:
                per = np.interp(angle, (220, 280), (0, 100))
                if per == 0 and dir == 0:
                    count += 0.5
                    dir = 1
                elif per == 100 and dir == 1:
                    count += 0.5
                    dir = 0

    # Check inactivity (no change in count for 10 seconds)
    if count > 0 and count != last_count:
        last_change_time = time.time()  # Reset timer **only if count has started updating**
        last_count = count  # Update last count

    # Ensure session doesn't end immediately
    if count > 0 and (time.time() - last_change_time > 10):
        return {"count": int(floor(count)), "final": True} # Indicate session end

    # Encode processed frame back to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode("utf-8")

    return {"count": int(floor(count)), "frame": frame_base64, "final": False}

@app.post("/reset_session/")
async def reset_session():
    global count, dir, last_change_time, last_count

    # Reset all variables
    count = 0
    dir = 0
    last_change_time = time.time()
    last_count = 0

    return {"message": "Session reset successfully"}

# Function to ping the API periodically
def keep_awake():
    try:
        response = requests.get(API_URL)
        print("Pinged API:", response.status_code)
    except Exception as e:
        print("Error in pinging API:", e)

# Background task to keep the API alive every 5 minutes
@app.on_event("startup")
async def schedule_keep_awake():
    while True:
        keep_awake()
        time.sleep(300)  # Wait for 5 minutes (300 seconds)
