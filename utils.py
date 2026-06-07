import numpy as np
import cv2

def calculate_ear(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) to detect eye closure.
    """
    try:
        # Vertical distances
        v1 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
        v2 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
        # Horizontal distance
        h = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
        ear = (v1 + v2) / (2.0 * h)
        return ear
    except:
        return 0.0

def get_roi_avg_color(frame, landmarks, landmark_indices):
    """
    Extract the average RGB color from a region defined by landmarks.
    """
    try:
        points = np.array([landmarks[idx] for idx in landmark_indices], dtype=np.int32)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        avg_color = cv2.mean(frame, mask=mask)[:3]
        return avg_color # BGR
    except:
        return (0, 0, 0)
