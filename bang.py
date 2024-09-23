import cv2
from ultralytics import YOLO  # Pastikan library YOLOv8 sudah terinstal


class ObjectDetection:
    def __init__(self, rtsp_url, start_x, start_y, end_x, end_y, entry_side='left'):
        # Inisialisasi dengan URL RTSP dari CCTV dan posisi garis
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url)

        # Posisi garis
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

        # Side determining entry ('left' or 'right' of the line)
        self.entry_side = entry_side
        self.entry_count = 0
        self.exit_count = 0

        # Memuat model YOLO
        self.model = self.load_model()

        # Store previous positions of objects to detect crossing
        self.previous_positions = {}

        # Cek apakah stream berhasil dibuka
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video stream from {self.rtsp_url}")

    def load_model(self):
        # Memuat model YOLOv8
        model = YOLO("yolov8x.pt")
        return model

    def draw_line(self, frame):
        # Menggambar garis pada frame menggunakan posisi yang diinisialisasikan
        cv2.line(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 0, 255), 2)

    def track_objects(self, frame):
        # Melacak objek menggunakan model YOLOv8
        results = self.model.track(frame, persist=True)
        return results

    def detect_crossing(self, object_id, prev_pos, current_pos):
        # Menghitung jarak dari titik objek ke garis
        def distance_from_line(x, y, x1, y1, x2, y2):
            # Rumus jarak titik ke garis
            return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / (
                    ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5)

        prev_dist = distance_from_line(prev_pos[0], prev_pos[1], self.start_x, self.start_y, self.end_x, self.end_y)
        curr_dist = distance_from_line(current_pos[0], current_pos[1], self.start_x, self.start_y, self.end_x, self.end_y)

        # Tentukan sisi objek terhadap garis
        prev_side = 'entry' if prev_pos[0] < self.start_x else 'exit'
        curr_side = 'entry' if current_pos[0] < self.start_x else 'exit'

        # Jika objek melewati garis (berpindah sisi)
        if prev_side != curr_side:
            if curr_side == self.entry_side:
                self.entry_count += 1
            else:
                self.exit_count += 1

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

            # Overlay for drawing
            overlay = frame.copy()

            # Menampilkan hasil deteksi objek pada frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Mengambil ID dan koordinat kotak pembatas
                    object_id = box.id
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Menghitung titik tengah objek
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Menggambar titik tengah
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Titik tengah berwarna hijau

                    # Lacak posisi sebelumnya dan deteksi apakah objek melewati garis
                    if object_id in self.previous_positions:
                        self.detect_crossing(object_id, self.previous_positions[object_id], (center_x, center_y))

                    # Simpan posisi saat ini
                    self.previous_positions[object_id] = (center_x, center_y)

            # Tampilkan teks entry dan exit
            cv2.putText(frame, f"Entry: {self.entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exit: {self.exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Tampilkan frame yang diambil dari CCTV
            cv2.imshow("CCTV Stream with Object Tracking", frame)

            # Untuk keluar, tekan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Setelah selesai, bebaskan resource
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rtsp_url = "rtsp://admin:AlphaBeta123@172.17.17.2:554/cam/realmonitor?channel=2&subtype=0"

    # Inisialisasi garis dengan posisi yang diinginkan
    start_x = 1000
    start_y = 100
    end_x = 120
    end_y = 480

    # Tambahkan sisi mana yang dianggap sebagai 'entry'
    entry_side = 'left'

    detector = ObjectDetection(rtsp_url, start_x, start_y, end_x, end_y, entry_side)
    detector()