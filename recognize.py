import cv2
import os
import pandas as pd
import datetime
import time
from collections import Counter


def preprocess_face(face):
    """Consistent preprocessing — must match capture and train."""
    face = cv2.resize(face, (200, 200))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    return face


def recognize_attendence():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_default.xml")
    TRAINER_PATH = os.path.join(BASE_DIR, "TrainingImageLabel", "Trainer.yml")
    STUDENT_CSV = os.path.join(BASE_DIR, "StudentDetails", "StudentDetails.csv")
    ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")

    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    if not os.path.exists(TRAINER_PATH):
        print(" Train images first.")
        return
    if not os.path.exists(STUDENT_CSV):
        print(" StudentDetails.csv not found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

    df = pd.read_csv(STUDENT_CSV)
    if 'Id' not in df.columns:
        df.columns = ['Id', 'Name']
    df['Id'] = df['Id'].astype(int)

    attendance = pd.DataFrame(columns=['Id', 'Name', 'Date', 'Time'])
    marked_ids = set()

    frame_predictions = {}   
    confidence_history = {} 
    last_confirmed = {}      

    cam = cv2.VideoCapture(0)

    CONFIDENCE_THRESHOLD = 65
    VOTE_FRAMES = 15          
    VOTE_MAJORITY = 10        
    MIN_FACE_SIZE = 80
    FRAME_SKIP = 2
    DETECTION_SCALE = 0.6

    frame_count = 0
    last_faces = []

    print("Starting Recognition... Press Q or ESC to quit.")
    print(f"   Threshold: {CONFIDENCE_THRESHOLD} | Votes needed: {VOTE_MAJORITY}/{VOTE_FRAMES}\n")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Smart Attendance", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            continue

        small = cv2.resize(img, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        detected = faceCascade.detectMultiScale(gray_small, 1.3, 5)

        if len(detected) > 0:
            faces = [(int(x/DETECTION_SCALE), int(y/DETECTION_SCALE),
                      int(w/DETECTION_SCALE), int(h/DETECTION_SCALE))
                     for (x, y, w, h) in detected]
            last_faces = faces
        else:
            faces = last_faces

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Track which face_keys are active this frame
        active_keys = set()

        for (x, y, w, h) in faces:

            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(img, "Move Closer", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                continue

            face_key = f"{x//60}_{y//60}"
            active_keys.add(face_key)

            face_roi = gray_full[y:y+h, x:x+w]
            face_roi = preprocess_face(face_roi)
            Id, conf = recognizer.predict(face_roi)

            if Id not in confidence_history:
                confidence_history[Id] = []
            confidence_history[Id].append(conf)

            recent_confs = confidence_history[Id][-20:]
            avg_conf = sum(recent_confs) / len(recent_confs)
            adaptive_threshold = CONFIDENCE_THRESHOLD - 5 if avg_conf < 40 else CONFIDENCE_THRESHOLD

            if face_key not in frame_predictions:
                frame_predictions[face_key] = []

            if conf < adaptive_threshold and Id in df['Id'].values:
                frame_predictions[face_key].append(Id)
            else:
                frame_predictions[face_key].append(-1)

            recent_votes = frame_predictions[face_key][-VOTE_FRAMES:]
            most_common_id, vote_count = Counter(recent_votes).most_common(1)[0]

            if len(recent_votes) >= VOTE_FRAMES and most_common_id != -1 and vote_count >= VOTE_MAJORITY:
                match = df.loc[df['Id'] == most_common_id]
                name = match['Name'].values[0].replace("_", " ")

                avg_display_conf = int(
                    sum(confidence_history.get(most_common_id, [conf])[-10:]) /
                    len(confidence_history.get(most_common_id, [conf])[-10:])
                )

                text = f"{name}  [{avg_display_conf}]"
                color = (0, 220, 0)

                last_confirmed[face_key] = (text, color)

                if most_common_id not in marked_ids:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    attendance.loc[len(attendance)] = [most_common_id, name, date, timeStamp]
                    marked_ids.add(most_common_id)
                    frame_predictions[face_key] = []  # reset votes after marking
                    print(f" Marked: {name} at {timeStamp}")

            elif face_key in last_confirmed:
                
                text, color = last_confirmed[face_key]

            else:
                votes_so_far = len(recent_votes)
                text = f"Identifying... {votes_so_far}/{VOTE_FRAMES}"
                color = (0, 165, 255)

            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(img, (x, y - th - 14), (x + tw + 8, y), color, -1)
            cv2.putText(img, text, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        stale_keys = set(last_confirmed.keys()) - active_keys
        for k in stale_keys:
            last_confirmed.pop(k, None)
            frame_predictions.pop(k, None)

        bar_h = 36
        overlay = img.copy()
        cv2.rectangle(overlay, (0, img.shape[0] - bar_h),
                      (img.shape[1], img.shape[0]), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        status = f"Marked: {len(marked_ids)} person(s)   |   Q / ESC to save & quit"
        cv2.putText(img, status, (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Smart Attendance", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    if not attendance.empty:
        file_name = f"Attendance_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        save_path = os.path.join(ATTENDANCE_DIR, file_name)
        attendance.to_csv(save_path, index=False)
        print(f"\n Attendance saved: {file_name}")
        print(attendance.to_string(index=False))
    else:
        print("\n⚠ No attendance marked this session.")

    cam.release()
    cv2.destroyAllWindows()