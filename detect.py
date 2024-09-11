import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import Counter

# Periksa apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# URL stream CCTV
url = "http://172.17.17.2/"

# Mengambil stream video dari URL CCTV
cap = cv2.VideoCapture(url)

# Mengatur resolusi stream CCTV (contoh: 640x480 untuk performa lebih cepat)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Memuat model YOLOv8
model = YOLO("yolov8n.pt")  # Pastikan model YOLOv8 telah di-download

# Fungsi untuk melakukan object detection
def detect_objects(frame):
    # Mengubah frame menjadi RGB (YOLOv8 memerlukan format RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Model inference
    results = model(rgb_frame)

    return results

if not cap.isOpened():
    print("Error: Tidak dapat mengakses stream CCTV.")
else:
    while True:
        # Membaca frame dari stream
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame dari CCTV.")
            break

        # Deteksi objek (real-time)
        results = detect_objects(frame)

        # Mendapatkan hasil deteksi
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].names

        # Menghitung jumlah objek per label
        detected_labels = [labels[int(label)] for label in results[0].boxes.cls.cpu().numpy()]
        label_count = Counter(detected_labels)

        # Gambar bounding boxes dan label pada frame
        for box, score, label in zip(boxes, scores, detected_labels):
            x1, y1, x2, y2 = [int(i) for i in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            text = f"{label}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Menampilkan jumlah objek yang terdeteksi di sudut frame
        y_offset = 20
        for label, count in label_count.items():
            text = f"{label}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

        # Menampilkan frame dengan bounding boxes dan object count
        cv2.imshow("CCTV Live Stream with Real-Time YOLOv8 Object Detection and Object Count", frame)

        # Keluar dengan menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Membersihkan jendela OpenCV dan mengakhiri stream
cap.release()
cv2.destroyAllWindows()
