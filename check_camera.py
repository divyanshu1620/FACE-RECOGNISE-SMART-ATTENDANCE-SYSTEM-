import cv2


def camer():
    print("Checking camera...")
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print(" Camera not found or already in use.")
        return

    print(" Camera opened successfully!")
    print("   Press Q to close the camera test.\n")

    frame_count = 0
    start = None

    import time
    start = time.time()

    while True:
        ret, img = cam.read()
        if not ret:
            print(" Failed to read frame from camera.")
            break

        frame_count += 1
        elapsed = time.time() - start
        fps = frame_count / elapsed if elapsed > 0 else 0

        h, w = img.shape[:2]

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        cv2.putText(img, f"Resolution: {w} x {h}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {fps:.1f}   |   Press Q to exit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        cv2.imshow("Camera Check", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
         break

    cam.release()
    cv2.destroyAllWindows()
    print(f" Camera test done. Avg FPS: {fps:.1f}")