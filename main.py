import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

# Load YOLO
model = YOLO('yolov8s.pt')

# Crossing line
x_line = 510

# Folder
save_path = "detected_people"
os.makedirs(save_path, exist_ok=True)

# Load COCO
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Webcam
cap = cv2.VideoCapture(0)


def process_frame():
    global cap
    ret, frame = cap.read()
    if not ret:
        return None, 0 # Captures the frame and returns none if it is not captured properly Ex. Error sa Webcam

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    detections = results[0].boxes.data.cpu().numpy()  # Ensure tensor is converted to NumPy

    if detections is None or len(detections) == 0:
        return frame, 0

    df = pd.DataFrame(detections).astype("float") # Convert to Panda

    person_count = 0
    for i, row in df.iterrows():
        if len(row) < 6:
            continue  # Skip invalid rows

        x1, y1, x2, y2, _, class_id = map(int, row)

        if 0 <= class_id < len(class_list) and class_list[class_id] == "person": # Confirm if person
            person_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Frame
            cv2.putText(frame, "person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Title

            # Save cropped image
            if x2 > x_line > x1:
                cropped_person = frame[max(y1, 0):min(y2, frame.shape[0]), max(x1, 0):min(x2, frame.shape[1])]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{save_path}/person_{timestamp}_{i}.jpg"
                cv2.imwrite(filename, cropped_person)
                print(f"Saved: {filename}")

    # Draw crossing line
    cv2.line(frame, (x_line, 0), (x_line, 500), (0, 0, 255), 2)

    # Common Area Label
    cv2.putText(frame, "<- Common Area", (x_line - 375, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Restricted Area Warning
    cv2.putText(frame, "Restricted Area ->", (x_line + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Your image may be captured & stored", (x_line + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)

    return frame, person_count


while True:
    frame, person_count = process_frame()
    if frame is None:
        break # Main Loop ng Detection

    # UI
    Color = (0, 0, 0)
    ui_frame = np.full((600, 1020, 3), Color, dtype=np.uint8)
    ui_frame[:500, :] = frame

    # Counter
    cv2.putText(ui_frame, f"People Count: {person_count}", (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("DETECTNCOUNT", ui_frame)

    # 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()