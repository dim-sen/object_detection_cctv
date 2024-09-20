import cv2
from ultralytics import YOLO  # Pastikan library YOLOv8 sudah terinstal

class ObjectDetection:
    def __init__(self, rtsp_url, start_x, start_y, end_x, end_y):
        # Inisialisasi dengan URL RTSP dari CCTV dan posisi garis
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)

        # Posisi garis
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

        # Memuat model YOLO
        self.model = self.load_model()

        # Cek apakah stream berhasil dibuka
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video stream from {self.rtsp_url}")

    def load_model(self):
        # Memuat model YOLOv8 dengan tracking
        model = YOLO("yolov8x.pt")  # Memuat model YOLOv8
        return model

    def draw_line(self, frame):
        # Menggambar garis pada frame menggunakan posisi yang diinisialisasikan
        cv2.line(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 255), 2)

    def track_objects(self, frame):
        # Melacak objek menggunakan model YOLOv8
        results = self.model.track(frame, persist=True)  # Tracking objek pada setiap frame
        return results

    def __call__(self):
        while True:
            # Membaca frame dari CCTV
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to retrieve frame. Exiting...")
                break

            # Melacak objek pada frame
            results = self.track_objects(frame)

            # Menggambar garis vertikal
            self.draw_line(frame)

            # Menampilkan hasil deteksi objek pada frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Menggambar kotak di sekitar objek yang dilacak
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Menampilkan ID objek yang dilacak (jika ada)
                    if hasattr(box, 'id'):
                        obj_id = int(box.id)
                        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Tampilkan frame yang diambil dari CCTV
            cv2.imshow("CCTV Stream with Object Tracking", frame)

            # Untuk keluar, tekan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Setelah selesai, bebaskan resource
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rtsp_url = "rtsp://admin:AlphaBeta123@172.17.17.2:554/cam/realmonitor?channel=4&subtype=0"

    # Inisialisasi garis dengan posisi yang diinginkan
    start_x = 500
    start_y = 300
    end_x = 120
    end_y = 480  # Ganti 480 dengan tinggi frame jika perlu

    detector = ObjectDetection(rtsp_url, start_x, start_y, end_x, end_y)
    detector()
