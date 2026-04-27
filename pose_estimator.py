# =============================================================================
#  pose_estimator.py — Head Movement Detection using YOLOv8n-pose
#
#  Uses: yolov8n-pose.pt (pretrained on COCO-Pose, 17 keypoints)
#
#  HOW IT WORKS (your contribution on top of pretrained model):
#    YOLOv8-pose gives 17 body keypoints per person.
#    We use only 5 face keypoints:
#      0 = Nose
#      1 = Left Eye
#      2 = Right Eye
#      3 = Left Ear
#      4 = Right Ear
#
#    YOUR CUSTOM HEAD DIRECTION ALGORITHM:
#
#    YAW  (left/right turn):
#      Compare horizontal distance of nose to left_eye vs nose to right_eye.
#      If nose is much closer to right eye → head turned LEFT (and vice versa).
#      Ratio = (nose_x - left_eye_x) / (right_eye_x - left_eye_x)
#      Ratio < 0.35 → LEFT,  Ratio > 0.65 → RIGHT
#
#    PITCH (up/down tilt):
#      Use ear visibility. When head tilts down, ears drop below eye level.
#      ear_mid_y vs eye_mid_y — if ears are significantly below → looking DOWN.
#      Also use nose_y relative to eye_mid_y for UP detection.
#
#    This is NOT provided by YOLOv8-pose — it only gives keypoint coordinates.
#    The direction logic is entirely your own algorithm.
# =============================================================================

from ultralytics import YOLO
import cv2
import numpy as np
import config


# COCO keypoint indices we use
NOSE       = 0
LEFT_EYE   = 1
RIGHT_EYE  = 2
LEFT_EAR   = 3
RIGHT_EAR  = 4


class HeadPoseEstimator:

    def __init__(self):
        print("[HeadPoseEstimator] Loading yolov8n-pose.pt ...")
        self.model = YOLO(config.YOLO_POSE_MODEL)
        print("[HeadPoseEstimator] Ready")

    def estimate(self, frame) -> dict:
        """
        Detect head direction from YOLOv8-pose keypoints.

        Returns:
            {
              "direction"  : "CENTER"|"LEFT"|"RIGHT"|"UP"|"DOWN"|"NO FACE"
              "yaw_ratio"  : float   — 0=far left, 1=far right (your formula)
              "pitch_ratio": float
              "is_alert"   : bool
              "alert_level": "OK" | "MEDIUM"
              "message"    : str
              "annotated"  : np.ndarray
            }
        """
        results = self.model(
            frame,
            conf=config.POSE_CONF,
            verbose=False
        )[0]

        annotated = results.plot()

        if results.keypoints is None or len(results.keypoints.data) == 0:
            return {
                "direction":   "NO FACE",
                "yaw_ratio":   0.5,
                "pitch_ratio": 0.5,
                "is_alert":    False,
                "alert_level": "OK",
                "message":     "No face detected",
                "annotated":   annotated,
            }

        # Use keypoints of the first (largest confidence) person
        kpts = results.keypoints.data[0]   # shape (17, 3) → x, y, confidence

        # Extract face keypoints
        def kp(idx):
            """Return (x, y, conf) for keypoint index."""
            return float(kpts[idx][0]), float(kpts[idx][1]), float(kpts[idx][2])

        nose_x,      nose_y,      nose_c      = kp(NOSE)
        left_eye_x,  left_eye_y,  left_eye_c  = kp(LEFT_EYE)
        right_eye_x, right_eye_y, right_eye_c = kp(RIGHT_EYE)
        left_ear_x,  left_ear_y,  left_ear_c  = kp(LEFT_EAR)
        right_ear_x, right_ear_y, right_ear_c = kp(RIGHT_EAR)

        direction   = "CENTER"
        yaw_ratio   = 0.5
        pitch_ratio = 0.5

        # ── YOUR YAW ALGORITHM ────────────────────────────────────────────────
        # Both eyes must be visible for a reliable yaw estimate
        if left_eye_c > 0.3 and right_eye_c > 0.3 and nose_c > 0.3:
            eye_span = right_eye_x - left_eye_x

            if abs(eye_span) > 5:    # eyes far enough apart to be meaningful
                # Where is the nose within the left-right eye span?
                yaw_ratio = (nose_x - left_eye_x) / eye_span
                yaw_ratio = float(np.clip(yaw_ratio, 0.0, 1.0))

                if yaw_ratio < (0.5 - config.HEAD_YAW_THRESH / 100):
                    direction = "LEFT"
                elif yaw_ratio > (0.5 + config.HEAD_YAW_THRESH / 100):
                    direction = "RIGHT"

        # ── YOUR PITCH ALGORITHM ──────────────────────────────────────────────
        # Use nose position relative to the eye midpoint
        if direction == "CENTER" and left_eye_c > 0.3 and right_eye_c > 0.3 and nose_c > 0.3:
            eye_mid_y   = (left_eye_y + right_eye_y) / 2.0
            eye_span_y  = abs(right_eye_y - left_eye_y) + 1   # avoid /0
            face_height = abs(nose_y - eye_mid_y)

            # Normalise pitch: how far is nose below/above eyes relative to face size
            pitch_ratio = (nose_y - eye_mid_y) / (face_height + 1e-6)
            pitch_ratio = float(np.clip((pitch_ratio + 1) / 2, 0.0, 1.0))

            if pitch_ratio > (0.5 + config.HEAD_PITCH_THRESH / 100):
                direction = "DOWN"
            elif pitch_ratio < (0.5 - config.HEAD_PITCH_THRESH / 100):
                direction = "UP"

        # ── Ear occlusion cross-check ─────────────────────────────────────────
        # If one ear disappears → head definitely turned that way
        if direction == "CENTER":
            if left_ear_c < 0.2 and right_ear_c > 0.3:
                direction = "RIGHT"   # left ear hidden → facing right
            elif right_ear_c < 0.2 and left_ear_c > 0.3:
                direction = "LEFT"    # right ear hidden → facing left

        is_alert = direction != "CENTER" and direction != "NO FACE"
        level    = "MEDIUM" if is_alert else "OK"
        message  = f"Head {direction}" if is_alert else "Head centered"

        # ── Draw direction label on annotated frame ───────────────────────────
        col   = config.COLOR_WARN if is_alert else config.COLOR_OK
        label = f"HEAD: {direction}"
        h, w  = annotated.shape[:2]
        cv2.putText(annotated, label,
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

        # Direction arrow overlay
        arrows = {
            "LEFT":  "<-- LOOKING LEFT",
            "RIGHT": "LOOKING RIGHT -->",
            "UP":    "^ LOOKING UP",
            "DOWN":  "v LOOKING DOWN",
        }
        if direction in arrows:
            cv2.putText(annotated, arrows[direction],
                        (10, h - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        config.COLOR_ALERT, 2)

        return {
            "direction":   direction,
            "yaw_ratio":   yaw_ratio,
            "pitch_ratio": pitch_ratio,
            "is_alert":    is_alert,
            "alert_level": level,
            "message":     message,
            "annotated":   annotated,
        }
