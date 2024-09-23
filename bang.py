import cv2
from ultralytics import YOLO


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

        # Simpan posisi tengah objek sebelumnya
        self.previous_centers = {}

        # Hitungan masuk dan keluar
        self.count_in = 0
        self.count_out = 0

    def load_model(self):
        # Memuat model YOLOv8
        model = YOLO("yolov8x.pt")  # Memuat model YOLOv8
        return model

    def draw_line(self, frame):
        # Menggambar garis pada frame menggunakan posisi yang diinisialisasikan
        cv2.line(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 255), 2)

    def track_objects(self, frame):
        # Melacak objek menggunakan model YOLOv8
        results = self.model.track(frame, persist=True)  # Tracking objek pada setiap frame
        return results

    def check_crossing(self, object_id, center_x):
        # Menghitung apakah objek melintasi garis vertikal
        previous_center = self.previous_centers.get(object_id, None)

        if previous_center is not None:
            prev_x = previous_center[0]

            # Deteksi arah lintasan (misalnya jika garis vertikal)
            if prev_x < self.start_x and center_x >= self.start_x:
                self.count_in += 1
                print(f"Object {object_id} masuk. Total masuk: {self.count_in}")
            elif prev_x >= self.start_x and center_x < self.start_x:
                self.count_out += 1
                print(f"Object {object_id} keluar. Total keluar: {self.count_out}")

        # Simpan posisi tengah objek saat ini untuk frame berikutnya
        self.previous_centers[object_id] = (center_x,)

    def display_counts(self, frame):
        # Menampilkan jumlah entry dan exit di pojok kiri atas
        entry_text = f"entry: {self.count_in}"
        exit_text = f"exit: {self.count_out}"

        # Posisi teks di pojok kiri atas
        cv2.putText(frame, entry_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, exit_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def __call__(self):
        while True:
            # Membaca frame dari CCTV
            ret, frame = self.cap.read()

            if not ret:
                print("Failed to retrieve frame. Exiting...")
                break

            # Melacak objek pada frame
            results = self.track_objects(frame)

            # Menggambar garis
            self.draw_line(frame)

            # Menampilkan hasil deteksi objek pada frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Mengambil koordinat kotak pembatas (bounding box)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    object_id = box.id  # Menggunakan ID objek untuk pelacakan

                    # Menghitung titik tengah (center point) dari kotak pembatas
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Cek apakah objek melewati garis
                    self.check_crossing(object_id, center_x)

                    # Menggambar titik tengah
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Titik tengah berwarna hijau

            # Tampilkan jumlah entry dan exit di pojok kiri atas
            self.display_counts(frame)

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
    start_x = 500  # Atur agar garis berada di tengah atau sesuai kebutuhan
    start_y = 100
    end_x = 500  # Posisi garis vertikal
    end_y = 480

    detector = ObjectDetection(rtsp_url, start_x, start_y, end_x, end_y)
    detector()