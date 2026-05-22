import cv2
import os
import numpy as np
from PIL import Image


def preprocess_face(face_np):
    """Consistent preprocessing — must match capture and recognize."""
    face_np = cv2.resize(face_np, (200, 200))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_np = clahe.apply(face_np)
    return face_np


def trainImages():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    training_image_path = os.path.join(BASE_DIR, "TrainingImage")
    training_label_path = os.path.join(BASE_DIR, "TrainingImageLabel")

    os.makedirs(training_label_path, exist_ok=True)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,       
        neighbors=8,   
        grid_x=8,       
        grid_y=8        
    )

    faces = []
    Ids = []
    skipped = 0

    print("🔄 Training Started...")

    image_files = [f for f in os.listdir(training_image_path) if f.endswith(".jpg")]

    if not image_files:
        print(" No images found in TrainingImage folder.")
        return

    for imageName in image_files:
        imagePath = os.path.join(training_image_path, imageName)

        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')

            imageNp = preprocess_face(imageNp)

            parts = imageName.split(".")
            Id = int(parts[1])

            faces.append(imageNp)
            Ids.append(Id)

        except Exception as e:
            print(f"⚠ Skipped {imageName}: {e}")
            skipped += 1
            continue

    if len(faces) == 0:
        print(" No valid images to train on.")
        return

    unique_ids = set(Ids)
    print(f" Training on {len(faces)} images across {len(unique_ids)} people...")

    from collections import Counter
    id_counts = Counter(Ids)
    for uid, count in id_counts.items():
        if count < 20:
            print(f"ID {uid} has only {count} samples — accuracy may be low. Recapture with 100+.")

    recognizer.train(faces, np.array(Ids))

    trainer_path = os.path.join(training_label_path, "Trainer.yml")
    recognizer.write(trainer_path)

    print(f"\n Training Completed!")
    print(f"   • Images trained : {len(faces)}")
    print(f"   • People         : {len(unique_ids)}")
    print(f"   • Skipped        : {skipped}")
    print(f"   • Model saved to : {trainer_path}")