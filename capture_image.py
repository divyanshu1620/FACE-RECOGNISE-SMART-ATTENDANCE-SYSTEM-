import os
import cv2
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

harcascadePath = os.path.join(BASE_DIR, "haarcascade_default.xml")
training_image_path = os.path.join(BASE_DIR, "TrainingImage")
student_details_path = os.path.join(BASE_DIR, "StudentDetails")

os.makedirs(training_image_path, exist_ok=True)
os.makedirs(student_details_path, exist_ok=True)

SAMPLES_NEEDED = 140  

STAGES = [
    "Look STRAIGHT at camera",
    "Turn slightly LEFT",
    "Turn slightly RIGHT",
    "Tilt HEAD UP slightly",
    "Tilt HEAD DOWN slightly",
    "Move to BRIGHTER area",
    "Move to DIMMER area",
]
SAMPLES_PER_STAGE = SAMPLES_NEEDED // len(STAGES)  


def preprocess_face(face):
    """Consistent preprocessing — must match train and recognize."""
    face = cv2.resize(face, (200, 200))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    return face


def takeImages():
    Id = input("Enter Your Id: ").strip()
    name = input("Enter Your Name: ").strip()

    if not Id.isdigit():
        print(" ID must be numeric")
        return

    if not all(c.isalpha() or c.isspace() for c in name):
        print("Name must contain alphabets only")
        return

    name = name.replace(" ", "_") 

    csv_path = os.path.join(student_details_path, "StudentDetails.csv")

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'Id' not in df.columns:
                df.columns = ['Id', 'Name']
        except Exception:
            df = pd.DataFrame(columns=["Id", "Name"])
    else:
        df = pd.DataFrame(columns=["Id", "Name"])

    if not df.empty:
        df['Id'] = df['Id'].astype(str)

    if Id in df['Id'].values:
        print("⚠ ID already exists! Use a different ID.")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(harcascadePath)

    sampleNum = 0
    MIN_FACE_SIZE = 80  

    print(f"\n📸 Capturing {SAMPLES_NEEDED} samples across {len(STAGES)} guided stages.")
    print("Follow the on-screen instructions. Press Q to quit early.\n")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        scale = 0.5
        small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray_small, 1.3, 5)

        faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale))
                 for (x, y, w, h) in faces]

        current_stage = min(sampleNum // SAMPLES_PER_STAGE, len(STAGES) - 1)
        stage_text = STAGES[current_stage]
        stage_progress = sampleNum % SAMPLES_PER_STAGE

        for (x, y, w, h) in faces:
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                cv2.putText(img, "Move Closer!", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                break

            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = gray_full[y:y+h, x:x+w]
            face = preprocess_face(face)

            sampleNum += 1
            file_name = f"{name}.{Id}.{sampleNum}.jpg"
            cv2.imwrite(os.path.join(training_image_path, file_name), face)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 2)
            break  # only largest face per frame

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        cv2.putText(img, f"Stage {current_stage+1}/{len(STAGES)}: {stage_text}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(img, f"Stage progress: {stage_progress}/{SAMPLES_PER_STAGE}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(img, f"Total: {sampleNum}/{SAMPLES_NEEDED}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Capturing Faces — Press Q to quit", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
          break
        elif sampleNum >= SAMPLES_NEEDED:
         break
        elif sampleNum >= SAMPLES_NEEDED:
            print(" All stages complete!")
            break

    cam.release()
    cv2.destroyAllWindows()

    if sampleNum > 0:
        df.loc[len(df)] = [Id, name]
        df.to_csv(csv_path, index=False)
        print(f" {sampleNum} images saved for {name} (ID: {Id})")
    else:
        print("⚠ No images captured. Check your camera or face position.")