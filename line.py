import math

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from numpy import linalg as LA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())


def line_vectorize(point1, point2):
    a = point2[0] - point1[0]
    b = point2[1] - point1[1]
    return [a, b]


def check_direction(pt1, pt2, pt3, pt4):
    u = np.array(line_vectorize(pt1, pt2))
    v = np.array(line_vectorize(pt3, pt4))
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
    if u[0] * v[1] - u[1] * v[0] < 0:
        return a
    else:
        return 360 - a
coordinateObject = []

class objekTracked:
    def __init__(self, objectId, x, y):
        self.objectId = objectId
        self.x = x
        self.y = y

def getObjectById(objectId):
    for object in coordinateObject:
        if object.objectId == objectId:
            return object
    return None

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
        model.to('cuda')
        print(model.device)
        model.fuse()
        return model

    def predict(self, img):
        results = self.model.track(img, stream=True)

        return results

    def plot_boxes(self, results, img):
        detections = []
        cv2.putText(img, f'Entry: {self.entry_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f'Exit: {self.exit_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                detections.append(([x1, y1, w, h], currentClass))
                print(box.id)
                if (getObjectById(box.id) != None):
                    self.is_line_crossed_danger(center_x, center_y, getObjectById(box.id).x,
                                                getObjectById(box.id).y)
                coordinateObject.append(objekTracked(box.id, center_x, center_y))
                img = r.plot()

        return detections, img

    def track_detect(self, detections, img, tracker):
        tracks = tracker.update_tracks(detections, frame=img)
        current_objects = set()

        # Hitung titik tengah dari garis utama
        mid_x = (self.start_x + self.end_x) // 2
        mid_y = (self.start_y + self.end_y) // 2

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = ltrb
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(y2), int(y2)

            # Hitung titik tengah dari bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Lacak objek
            current_objects.add(track_id)

        #     # Periksa jika objek sudah pernah terdeteksi sebelumnya
        #     if track_id in self.objects_crossed:
        #         prev_center = self.objects_crossed[track_id]['position']
        #         prev_direction = self.objects_crossed[track_id]['direction']
        #
        #         #
        #         # # Hitung vektor dari posisi sebelumnya ke posisi sekarang
        #         # direction_vector = (center_x - prev_center[0], center_y - prev_center[1])
        #         #
        #         # # Hitung vektor dari titik tengah garis utama ke titik ke-3 (acuan entry)
        #         # reference_vector = (self.reference_x - mid_x, self.reference_y - mid_y)
        #         #
        #         # # Periksa jika objek melewati garis utama
        #         # if (prev_center[0] < mid_x and center_x >= mid_x) or (prev_center[0] >= mid_x and center_x < mid_x):
        #         #     # Periksa arah pergerakan: mendekati atau menjauhi titik ke-3
        #         #     if self.is_moving_towards(direction_vector, reference_vector):
        #         #         print(f"Object {track_id} entered.")
        #         #         self.entry_count += 1
        #         #     else:
        #         #         print(f"Object {track_id} exited.")
        #         #         self.exit_count += 1
        #
        #         # Update posisi dan arah terbaru objek
        #         self.objects_crossed[track_id] = {'position': (center_x, center_y), 'direction': direction_vector}
        #     else:
        #         # Simpan posisi awal objek
        #         self.objects_crossed[track_id] = {'position': (center_x, center_y), 'direction': (0, 0)}
        #
        #     # Hapus objek yang hilang dari pelacakan
        # for track_id in list(self.objects_crossed.keys()):
        #     if track_id not in current_objects:
        #         del self.objects_crossed[track_id]

        # Tampilkan jumlah objek yang masuk dan keluar
        cv2.putText(img, f'Entry: {self.entry_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f'Exit: {self.exit_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return img

    def is_moving_towards(self, direction_vector, reference_vector):
        """
        Menghitung apakah vektor arah objek mendekati atau menjauhi titik ke-3
        menggunakan dot product.
        """
        dot_product = direction_vector[0] * reference_vector[0] + direction_vector[1] * reference_vector[1]
        return dot_product > 0  # Positif berarti objek mendekati titik ke-3, negatif berarti menjauhi

    def draw_lines(self, img):
        # Gambar garis utama
        if self.start_x and self.start_y and self.end_x and self.end_y:
            cv2.line(img, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 255), 2)

        # Hitung titik tengah dari garis utama
        mid_x = (self.start_x + self.end_x) // 2
        mid_y = (self.start_y + self.end_y) // 2

        # Gambar garis dari titik tengah ke titik ke-3 (reference)
        if self.reference_x and self.reference_y:
            cv2.line(img, (mid_x, mid_y), (self.reference_x, self.reference_y), (0, 255, 0), 2)

        return img

    def is_line_crossed_danger(self, cx, cy, prev_cx, prev_cy):
        p3 = (self.start_x, self.start_y)
        p4 = (self.end_x, self.end_y)
        p1 = (cx, cy)
        p2 = (prev_cx, prev_cy)
        tc1 = (p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])
        tc2 = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        td1 = (p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])
        td2 = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])

        if (tc2 != 0 and td2 != 0 and 0 <= tc1 / tc2 <= 1 and 0 <= td1 / td2 <= 1):
            self.entry_count = self.entry_count + 1
            print(self.entry_count)

    def check_direction(pt1, pt2, pt3, pt4):
        u = np.array(line_vectorize(pt1, pt2))
        v = np.array(line_vectorize(pt3, pt4))
        i = np.inner(u, v)
        n = LA.norm(u) * LA.norm(v)
        c = i / n
        a = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        if u[0] * v[1] - u[1] * v[0] < 0:
            return a
        else:
            return 360 - a

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
            # detect_frame = self.track_detect(detections, frames, tracker)
            detect_frame = self.draw_lines(frames)
            cv2.imshow('Image', frames)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Ganti URL RTSP dengan URL CCTV kamu
rtsp_url = "rtsp://admin:AlphaBeta123@172.17.17.2:554/cam/realmonitor?channel=4&subtype=0"
detector = ObjectDetection(capture=rtsp_url)
detector()
