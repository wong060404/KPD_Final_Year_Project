#public_variable.py
import cv2
from pathlib import Path

# basic path
BASE_DIR = Path("/Users/alex_wcc/PycharmProjects/PythonProject")

# path of raw face
IMAGES_DIR = BASE_DIR / "Images"

# path of processed face
PROCESSED_DIR = BASE_DIR / "Images_processed"

# data after trained
MODEL_PATH = BASE_DIR / "model.yml"

# (name <-> id)
LABELS_PATH = BASE_DIR / "labels.json"

# default camera idex (0) --> 1 for other camera
CAM_INDEX = 1

# face recognize index
IMG_SIZE = (200, 200)  # face resize size
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 辨識參數
RECOG_INTERVAL_FRAMES = 10   # times per frames
CONFIDENCE_THRESHOLD = 70    # confidence (LBPH: 越小越好)

# Check time and true times required
WINDOW_TIME = 4.0           # Time required (in Seconds)
REQUIRED_COUNT = 10         # "True" times required in limited time
