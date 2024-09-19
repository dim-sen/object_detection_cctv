import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url = "rtsp://admin:AlphaBeta123@172.17.17.2:554/cam/realmonitor?channel=4&subtype=0"

cap = cv2.VideoCapture(url)

model = YOLO("yolov8n.pt")

score_threshold = 0.5

def detect_objects(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(rgb_frame)

    return results

if not cap.isOpened():
    print("Error: Tidak dapat mengakses stream CCTV.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame dari CCTV.")
            break

        results = detect_objects(frame)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].names

        detected_labels = [labels[int(label)] for label in results[0].boxes.cls.cpu().numpy()]
        label_count = Counter([label for label, score in zip(detected_labels, scores) if score >= score_threshold])

        for box, score, label in zip(boxes, scores, detected_labels):
            x1, y1, x2, y2 = [int(i) for i in box]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            text = f"{label}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        y_offset = 30
        for label, count in label_count.items():
            text = f"{label}: {count}"

            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(frame, (10, y_offset - 30), (10 + text_width, y_offset), (0, 0, 0), -1)

            cv2.putText(frame, text, (10, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            y_offset += text_height + 20  # Menambah jarak antar baris

        cv2.imshow("CCTV Live Stream with Real-Time YOLOv8 Object Detection and Object Count", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()