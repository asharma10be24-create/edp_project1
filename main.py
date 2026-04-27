import cv2
import time
import threading
import queue

import config
import logger
from detector import ObjectDetector
from pose_estimator import HeadPoseEstimator


# ================= CAMERA =================
class CameraStream:
    def __init__(self):
        # ✅ FIX: Initialize camera properly (Mac compatible)
        self.cap = cv2.VideoCapture(config.INPUT_SOURCE)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open camera")

        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()


# ================= YOLO WORKER =================
class YOLOWorker:
    def __init__(self, detector):
        self.detector = detector
        self.q = queue.Queue(maxsize=1)
        self.result = None
        self.lock = threading.Lock()

        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            frame = self.q.get()
            res = self.detector.detect(frame)
            with self.lock:
                self.result = res

    def submit(self, frame):
        if not self.q.full():
            self.q.put(frame)

    def get(self):
        with self.lock:
            return self.result


# ================= MAIN =================
def main():
    print("Starting System...")

    logger.init()

    detector = ObjectDetector()
    pose = HeadPoseEstimator()

    cam = CameraStream()
    yolo = YOLOWorker(detector)

    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_count += 1

        # Run detection
        if frame_count % config.YOLO_EVERY == 0:
            yolo.submit(frame)

        detect_res = yolo.get()
        if detect_res is None:
            detect_res = {
                "annotated": frame,
                "person_count": 0,
                "alert_objects": [],
                "is_alert": False,
            }

        # Run pose
        pose_res = pose.estimate(frame)

        display = detect_res["annotated"]

        # Show
        cv2.imshow("Proctoring System", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()