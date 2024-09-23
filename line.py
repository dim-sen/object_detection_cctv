import math

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, capture):
        self.capture = capture
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        # Posisi garis
        self.start_x, self.start_y = 0, 0  # Titik 1
        self.end_x, self.end_y = 0, 0      # Titik 2
        self.reference_x, self.reference_y = 0, 0  # Titik 3
        self.point_count = 0
        self.objects_crossed = {}
        self.entry_count = 0
        self.exit_count = 0
        self.object_positions = {}  # Menyimpan posisi terakhir objek

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
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]
                conf = math.ceil(box.conf[0] * 100) / 100
                if conf > 0.5:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
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
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_objects.add(track_id)

            # Gambar ekor untuk menunjukkan arah objek
            if track_id in self.object_positions:
                prev_x, prev_y = self.object_positions[track_id]
                cv2.line(img, (prev_x, prev_y), (center_x, center_y), (255, 0, 0), 2)  # Ekor berwarna biru

            self.object_positions[track_id] = (center_x, center_y)

            # Menghitung arah berdasarkan garis utama dan titik ke-3
            if track_id in self.objects_crossed:
                prev_x = self.objects_crossed[track_id]

                # Periksa jika objek melewati garis utama
                if (prev_x < self.start_x and center_x >= self.start_x):
                    if center_x > self.reference_x:  # Menuju ke titik ke-3
                        print(f"Object {track_id} entered.")
                        self.entry_count += 1
                    else:  # Menjauhi titik ke-3
                        print(f"Object {track_id} exited.")
                        self.exit_count += 1
                elif (prev_x >= self.start_x and center_x < self.start_x):
                    if center_x < self.reference_x:  # Menuju ke titik ke-3
                        print(f"Object {track_id} entered.")
                        self.entry_count += 1
                    else:  # Menjauhi titik ke-3
                        print(f"Object {track_id} exited.")
                        self.exit_count += 1

                self.objects_crossed[track_id] = center_x
            else:
                self.objects_crossed[track_id] = center_x

        for track_id in list(self.objects_crossed.keys()):
            if track_id not in current_objects:
                del self.objects_crossed[track_id]
                del self.object_positions[track_id]  # Hapus posisi objek yang hilang

        cv2.putText(img, f'Entry: {self.entry_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f'Exit: {self.exit_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img

    def draw_lines(self, img):
        # Gambar garis utama
        if self.start_x and self.start_y and self.end_x and self.end_y:
            cv2.line(img, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 255), 2)

        # Gambar garis penentu entry/exit
        if self.end_x and self.end_y and self.reference_x and self.reference_y:
            cv2.line(img, (self.end_x, self.end_y), (self.reference_x, self.reference_y), (0, 255, 0), 2)

        return img

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.point_count == 0:
                self.start_x, self.start_y = x, y  # Titik 1
                self.point_count += 1
            elif self.point_count == 1:
                self.end_x, self.end_y = x, y  # Titik 2
                self.point_count += 1
            elif self.point_count == 2:
                self.reference_x, self.reference_y = x, y  # Titik 3
                self.point_count += 1
                print(f"Lines set: Start({self.start_x}, {self.start_y}), End({self.end_x}, {self.end_y}), Reference({self.reference_x}, {self.reference_y})")

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        if not cap.isOpened():
            print("Error: Unable to open video capture.")
            return

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)

        tracker = DeepSort(max_age=15, n_init=3, nms_max_overlap=1.0,
                           max_cosine_distance=0.3, nn_budget=None,
                           embedder="mobilenet", half=True,
                           bgr=True, embedder_gpu=True)

        while True:
            ret, img = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            results = self.predict(img)
            detections, frames = self.plot_boxes(results, img)
            detect_frame = self.track_detect(detections, frames, tracker)
            detect_frame = self.draw_lines(detect_frame)

            cv2.imshow('Image', detect_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Ganti URL RTSP dengan URL CCTV kamu
rtsp_url = "rtsp://admin:AlphaBeta123@172.17.17.2:554/cam/realmonitor?channel=3&subtype=0"
detector = ObjectDetection(capture=rtsp_url)
detector()