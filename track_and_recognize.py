#track_and_recognize.py
import cv2
import time
import json
from pathlib import Path
import numpy as np
from collections import defaultdict, deque

#import variable
from public_variable import CAM_INDEX, MODEL_PATH, LABELS_PATH, IMG_SIZE, CASCADE, RECOG_INTERVAL_FRAMES, CONFIDENCE_THRESHOLD, IMAGES_DIR, PROCESSED_DIR, WINDOW_TIME, REQUIRED_COUNT


#MODEL_PATH = Path("/Users/alex_wcc/PycharmProjects/PythonProject/model.yml")
#LABELS_PATH = Path("/Users/alex_wcc/PycharmProjects/PythonProject/labels.json")
#CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#IMG_SIZE = (200, 200)
#RECOG_INTERVAL_FRAMES = 10  # Second pre frame 
#CONFIDENCE_THRESHOLD = 70   # confidence from LBPH (lower index = higher accuracy)
#WINDOW_TIME = 4.0           # Time required (in Seconds)
#REQUIRED_COUNT = 10         # "True" times required in limited time

def load_model_and_labels():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("model.yml 或 labels.json not excess, please run process_and_train.py first!")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    with open(LABELS_PATH, "r") as fh:
        label_map = json.load(fh)  # name -> id
    # invert map
    id2name = {v:k for k,v in label_map.items()}
    return recognizer, id2name

class TrackedFace:
    def __init__(self, tracker, bbox, start_frame):
        self.tracker = tracker
        self.bbox = bbox  # (x,y,w,h)
        self.last_seen = start_frame
        self.frames_tracked = 0
        self.last_result = None  # (recognized_bool, name_or_None, confidence)
        self.id = int(time.time() * 1000)  # id

def run_tracking(cam_index=1):
    recognizer, id2name = load_model_and_labels()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot turn on the camera")

    tracked = {}  # id -> TrackedFace
    frame_count = 0

    # Use deque to store everyone's True times
    recognition_times = defaultdict(lambda: deque())

    print("Start tracking! Press q to end!")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        display = frame.copy()

        # update all tracker
        remove_ids = []
        for tid, tf in list(tracked.items()):
            ok, bbox = tf.tracker.update(frame)
            if not ok:
                remove_ids.append(tid)
            else:
                x,y,w,h = [int(v) for v in bbox]
                tf.bbox = (x,y,w,h)
                tf.frames_tracked += 1
                tf.last_seen = frame_count

                # Blue retangle (keep tracing)
                color = (255,0,0)  # BGR: blue

                # recognise every few frames
                if frame_count % RECOG_INTERVAL_FRAMES == 0:
                    # extract ROI, preprocess
                    face = frame[y:y+h, x:x+w]
                    if face.size > 0:
                        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        face_proc = cv2.resize(gray, IMG_SIZE)
                        label_id, confidence = recognizer.predict(face_proc)
                        # LBPH: lower confidence = better match
                        if confidence < CONFIDENCE_THRESHOLD:
                            name = id2name.get(label_id, "Unknown")
                            color = (0,255,0)  # green
                            recognized = True
                        else:
                            name = None
                            color = (0,0,255)  # red
                            recognized = False
                        tf.last_result = (recognized, name, float(confidence))
                        # output (or callback) || print the output
                        print(f"[Frame {frame_count}] Tracker {tid} recognition -> {tf.last_result}")

                        #Accumulate 10 True responses from the same person within {} seconds
                        if recognized and name is not None:
                            now = time.time()
                            dq = recognition_times[name]
                            dq.append(now)
                            # remove WINDOW_TIME 's previous record which is outrange
                            while dq and now - dq[0] > WINDOW_TIME:
                                dq.popleft()
                            if len(dq) >= REQUIRED_COUNT:
                                print(f"This is{name}")
                                dq.clear()  # clear all

                # draw box and label
                cv2.rectangle(display, (x,y), (x+w, y+h), color, 2)
                # label text
                label_text = ""
                if tf.last_result is not None and tf.last_result[0]:
                    label_text = f"{tf.last_result[1]} ({tf.last_result[2]:.1f})"
                elif tf.last_result is not None:
                    label_text = f"Unknown ({tf.last_result[2]:.1f})"
                else:
                    label_text = "Tracking"
                cv2.putText(display, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # remove lost trackers
        for rid in remove_ids:
            tracked.pop(rid, None)

        # Perform detection every few frames to find new faces (to avoid running detect on every frame)
        if frame_count % 5 == 0:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
            # 將每個 detect 到的 face 嘗試與已存在的 tracker 做 IoU 比對，如果沒有重疊則新增 tracker
            for (x,y,w,h) in faces:
                found_overlap = False
                for tf in tracked.values():
                    tx,ty,tw,th = tf.bbox
                    # compute IoU
                    ix1 = max(x, tx); iy1 = max(y, ty)
                    ix2 = min(x+w, tx+tw); iy2 = min(y+h, ty+th)
                    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                    inter = iw * ih
                    union = (w*h) + (tw*th) - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.3:
                        found_overlap = True
                        break
                if not found_overlap:
                    # create tracker for this face region
                    #tracker = cv2.TrackerMOSSE_create()
                    tracker = cv2.legacy.TrackerMOSSE_create()

                    tracker.init(frame, (x,y,w,h))
                    tf = TrackedFace(tracker, (x,y,w,h), frame_count)
                    tracked[tf.id] = tf

        cv2.imshow("Tracking & Recognition (q to quit)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finish tracing!")
    # can print something more detail here


    return tracked

if __name__ == "__main__":
    run_tracking()

