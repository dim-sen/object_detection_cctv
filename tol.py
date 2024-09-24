import cv2
import numpy as np
import torch
from ultralytics import YOLO
from numpy import linalg as LA
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def line_vectorize(point1, point2):
    a = point2[0] - point1[0]
    b = point2[1] - point1[1]
    return [a, b]


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
        self.start_x, self.start_y = 0, 0
        self.end_x, self.end_y = 0, 0
        self.reference_x, self.reference_y = 0, 0
        self.point_count = 0
        self.entry_count = 0
        self.exit_count = 0
        self.running = True  # To control the thread

    def load_model(self):
        model = YOLO("yolov5s.pt")
        model.to('cuda')
        model.fuse()
        return model

    def predict(self, img):
        results = self.model.track(img, persist=True)
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
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                print(box.id)
                if getObjectById(box.id) is not None:
                    self.is_line_crossed_danger(center_x, center_y, getObjectById(box.id).x, getObjectById(box.id).y)
                else:
                    coordinateObject.append(objekTracked(box.id, center_x, center_y))

                # Update position if already tracked
                for i, item in enumerate(coordinateObject):
                    if item.objectId == box.id:
                        coordinateObject[i] = objekTracked(box.id, center_x, center_y)
                        break
                img = r.plot()

        return detections, img

    def draw_lines(self, img):
        if self.start_x and self.start_y and self.end_x and self.end_y:
            cv2.line(img, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 255), 2)
        mid_x = (self.start_x + self.end_x) // 2
        mid_y = (self.start_y + self.end_y) // 2
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
        if tc2 != 0 and td2 != 0 and 0 <= tc1 / tc2 <= 1 and 0 <= td1 / td2 <= 1:
            direction = self.check_direction(p1, p2, p3, p4)
            if direction < 180:
                self.entry_count += 1
            if direction > 180:
                self.exit_count += 1

    def check_direction(self, pt1, pt2, pt3, pt4):
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
                self.start_x, self.start_y = x, y
                self.point_count += 1
            elif self.point_count == 1:
                self.end_x, self.end_y = x, y
                self.point_count += 1
            elif self.point_count == 2:
                self.reference_x, self.reference_y = x, y
                self.point_count += 1
                print(
                    f"Lines set: Start({self.start_x}, {self.start_y}), End({self.end_x}, {self.end_y}), Reference({self.reference_x}, {self.reference_y})")

    def process_frames(self):
        cap = cv2.VideoCapture(self.capture)
        if not cap.isOpened():
            print("Error: Unable to open video capture.")
            return

        while self.running:
            ret, img = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            results = self.predict(img)
            detections, img = self.plot_boxes(results, img)
            img = self.draw_lines(img)
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        cap.release()
        cv2.destroyAllWindows()

    def __call__(self):
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.mouse_callback)

        # Start the frame processing thread
        thread = threading.Thread(target=self.process_frames)
        thread.start()

        # Join the thread back to the main thread to wait for its completion
        thread.join()


# Ganti URL RTSP dengan URL CCTV kamu
rtsp_url = "https://www.tjt-info.co.id/LiveApp/streams/998223146655371157400972.m3u8"
detector = ObjectDetection(capture=rtsp_url)
detector()