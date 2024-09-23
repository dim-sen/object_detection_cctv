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
        self.count_in = {}
        self.count_out = {}

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

    def check_crossing(self, object_id, center_x, label):
        # Menghitung apakah objek melintasi garis vertikal
        previous_center = self.previous_centers.get(object_id, None)

        if previous_center is not None:
            prev_x = previous_center[0]

            # Deteksi arah lintasan (misalnya jika garis vertikal)
            if prev_x < self.start_x and center_x >= self.start_x:
                # Jika object belum pernah ada, tambahkan ke hitungan
                if label not in self.count_in:
                    self.count_in[label] = 0
                self.count_in[label] += 1
                print(f"{label} masuk. Total masuk: {self.count_in[label]}")
            elif prev_x >= self.start_x and center_x < self.start_x:
                if label not in self.count_out:
                    self.count_out[label] = 0
                self.count_out[label] += 1
                print(f"{label} keluar. Total keluar: {self.count_out[label]}")

        # Simpan posisi tengah objek saat ini untuk frame berikutnya
        self.previous_centers[object_id] = (center_x,)

    def draw_entry_exit_info(self, frame):
        # Menampilkan informasi 'entry' dan 'exit' di frame
        entry_text = "Entry:"
        exit_text = "Exit:"

        # Tampilkan jumlah masuk (entry)
        for label, count in self.count_in.items():
            entry_text += f"\n{label}: {count}"

        # Tampilkan jumlah keluar (exit)
        for label, count in self.count_out.items():
            exit_text += f"\n{label}: {count}"

        # Atur teks pada pojok kanan atas untuk 'entry'
        y_offset = 20
        for i, line in enumerate(entry_text.split('\n')):
            cv2.putText(frame, line, (frame.shape[1] - 250, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Atur teks pada pojok kiri atas untuk 'exit'
        y_offset = 20
        for i, line in enumerate(exit_text.split('\n')):
            cv2.putText(frame, line, (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

            overlay = frame.copy()

            # Menampilkan hasil deteksi objek pada frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Mengambil koordinat kotak pembatas (bounding box)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    object_id = box.id  # Menggunakan ID objek untuk pelacakan
                    label = box.cls  # Menggunakan label objek (misalnya "person", "car")

                    # Menghitung titik tengah (center point) dari kotak pembatas
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Cek apakah objek melewati garis
                    self.check_crossing(object_id, center_x, label)

                    # Menggambar titik tengah
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Titik tengah berwarna hijau

            # Tampilkan informasi entry dan exit
            self.draw_entry_exit_info(frame)

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