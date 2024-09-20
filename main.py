from ultralytics import YOLO
import cv2
import cvzone
import math
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectDetection:
    def __init__(self, capture):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.gate_x = 300  # Posisi garis vertikal
        self.objects_crossed = {}  # Untuk melacak objek dan arah lintasan
        self.entry_count = 0
        self.exit_count = 0

    def load_model(self):
        model = YOLO("yolov5s.pt")
        model.fuse()
        return model

    def predict(self, img):
        results = self.model(img, stream=True)
        return results

    def plot_boxes(self, results, img):
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(f"Box coordinates: ({x1}, {y1}), ({x2}, {y2})")

                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]
                conf = math.ceil(box.conf[0] * 100) / 100

                if conf > 0.5:
                    print(f"Detected {currentClass} with confidence {conf}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{currentClass} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    detections.append(([x1, y1, w, h], conf, currentClass))

        return detections, img

    def track_detect(self, detections, img, tracker):
        tracks = tracker.update_tracks(detections, frame=img)
        current_objects = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = ltrb
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Hitung titik tengah
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Gambar kotak, ID, dan titik tengah
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
            cv2.putText(img, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)  # Gambar titik tengah

            # Lacak objek
            current_objects.add(track_id)

            # Periksa jika titik tengah objek melintasi garis vertikal
            if track_id in self.objects_crossed:
                prev_x = self.objects_crossed[track_id]
                if (prev_x < self.gate_x and center_x >= self.gate_x):
                    print(f"Object {track_id} entered.")
                    self.entry_count += 1
                elif (prev_x >= self.gate_x and center_x < self.gate_x):
                    print(f"Object {track_id} exited.")
                    self.exit_count += 1
                self.objects_crossed[track_id] = center_x
            else:
                self.objects_crossed[track_id] = center_x

        # Hapus objek yang hilang dari pelacakan
        for track_id in list(self.objects_crossed.keys()):
            if track_id not in current_objects:
                del self.objects_crossed[track_id]

        # Menampilkan jumlah objek yang masuk dan keluar
        cv2.putText(img, f'Entry: {self.entry_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f'Exit: {self.exit_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img

    def draw_vertical_line(self, img, x_position):
        # Gambar garis vertikal
        cv2.line(img, (x_position, 0), (x_position, img.shape[0]), (0, 0, 255), 2)
        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        if not cap.isOpened():
            print("Error: Unable to open video capture.")
            return

        tracker = DeepSort(max_age=15,  # Nilai lebih tinggi untuk toleransi deteksi yang hilang
                           n_init=3,  # Diperlukan beberapa deteksi sebelum pelacakan dimulai
                           nms_max_overlap=1.0,
                           max_cosine_distance=0.3,
                           nn_budget=None,
                           override_track_class=None,
                           embedder="mobilenet",
                           half=True,
                           bgr=True,
                           embedder_gpu=True,
                           embedder_model_name=None,
                           embedder_wts=None,
                           polygon=False,
                           today=None)

        while True:
            ret, img = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames, tracker)
            detect_frame = self.draw_vertical_line(detect_frame, self.gate_x)

            cv2.imshow('Image', detect_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Ganti URL RTSP dengan URL CCTV kamu
rtsp_url = "rtsp://admin:AlphaBeta123@172.17.17.2:554/cam/realmonitor?channel=3&subtype=0"
detector = ObjectDetection(capture=rtsp_url)
detector()
