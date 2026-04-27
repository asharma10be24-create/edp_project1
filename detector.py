# =============================================================================
#  detector.py — Object & Multi-Person Detection using YOLOv8n
#
#  Uses: yolov8n.pt (pretrained on COCO)
#  Detects:
#    - Cell phone, book, laptop, remote  → HIGH alert
#    - Person count > MAX_PERSONS        → HIGH alert
#
#  Returns a result dict every call to .detect(frame)
# =============================================================================

from ultralytics import YOLO
import cv2
import config


class ObjectDetector:
    """
    Wraps YOLOv8n for object + person detection.
    Loaded once at startup — .detect() called every Nth frame.
    """

    def __init__(self):
        print("[ObjectDetector] Loading yolov8n.pt ...")
        self.model = YOLO(config.YOLO_DETECT_MODEL)
        print("[ObjectDetector] Ready")

    def detect(self, frame) -> dict:
        """
        Run inference on frame.

        Returns:
            {
              "annotated"    : np.ndarray  — frame with boxes drawn
              "person_count" : int
              "alert_objects": list[str]   — e.g. ["cell phone(87%)", "book(62%)"]
              "is_alert"     : bool
              "alert_level"  : "OK" | "HIGH"
              "message"      : str
            }
        """
        results = self.model(
            frame,
            conf=config.DETECT_CONF,
            verbose=False
        )[0]

        person_count  = 0
        alert_objects = []

        if results.boxes is not None:
            for box in results.boxes:
                cls   = int(box.cls[0])
                label = self.model.names[cls]
                conf  = float(box.conf[0])

                if label == "person":
                    person_count += 1

                if label in config.ALERT_OBJECTS:
                    alert_objects.append(f"{label} ({conf:.0%})")

        # Annotated frame (boxes drawn by ultralytics)
        annotated = results.plot()

        is_alert = bool(alert_objects) or person_count > config.MAX_PERSONS

        if person_count > config.MAX_PERSONS:
            level   = "HIGH"
            message = f"Multiple persons: {person_count}"
        elif alert_objects:
            level   = "HIGH"
            message = f"Object detected: {', '.join(alert_objects)}"
        else:
            level   = "OK"
            message = f"Person: {person_count}  Objects: none"

        return {
            "annotated":     annotated,
            "person_count":  person_count,
            "alert_objects": alert_objects,
            "is_alert":      is_alert,
            "alert_level":   level,
            "message":       message,
        }
