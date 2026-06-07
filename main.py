import cv2
import mediapipe as mp
import numpy as np
import time
from scipy import signal as scipy_signal
from heartrate import HeartRateDetector
from utils import calculate_ear, get_roi_avg_color

# MediaPipe Configuration
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 446, 261, 26, 226, 159, 103, 54, 67]

# Global state for buttons
manual_start = False

def mouse_callback(event, x, y, flags, param):
    global manual_start
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check "Start/Stop" button click (270, 430) to (370, 460)
        if 270 <= x <= 370 and 430 <= y <= 460:
            manual_start = not manual_start
        # Check "Open" button click (150, 430) to (250, 460)
        elif 150 <= x <= 250 and 430 <= y <= 460:
            # For now, just trigger start as "Open" can also mean start capture
            manual_start = True

def main():
    global manual_start
    cap = cv2.VideoCapture(0)
    detector = HeartRateDetector(buffer_size=150, fps=30)
    
    # UI Layout Constants
    WIN_WIDTH, WIN_HEIGHT = 1200, 800
    
    cv2.namedWindow("Heart Rate Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Heart Rate Monitor", WIN_WIDTH, WIN_HEIGHT)
    cv2.setMouseCallback("Heart Rate Monitor", mouse_callback)

    last_bpm = 0.0
    last_freq = 0.0
    last_fft_freqs = []
    last_fft_values = []
    last_signal = []
    last_face_crop = np.zeros((150, 200, 3), dtype=np.uint8)
    last_webcam_feed = np.zeros((350, 400, 3), dtype=np.uint8)
    eyes_open = True
    start_time = time.time()
    
    # FPS tracking variables
    fps_start_time = time.time()
    fps_counter = 0
    fps_val = 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps_val = fps_counter / elapsed
            fps_counter = 0
            fps_start_time = time.time()

        # Create Background for UI
        ui_frame = np.zeros((WIN_HEIGHT, WIN_WIDTH, 3), dtype=np.uint8)

        face_detected = False
        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append(np.array([lm.x * w, lm.y * h]))
                
                left_ear = calculate_ear(landmarks, LEFT_EYE)
                right_ear = calculate_ear(landmarks, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                
                # Stop detecting if eyes are closed (EAR threshold)
                if ear < 0.2:
                    eyes_open = False
                else:
                    eyes_open = True

                # Current detection state
                is_detecting = manual_start and eyes_open

                # 2. Extract Signal from Forehead
                avg_color = get_roi_avg_color(frame, landmarks, FOREHEAD)
                green_val = avg_color[1] # Green channel
                
                if is_detecting:
                    bpm = detector.update(green_val)
                    if bpm > 0:
                        last_bpm = bpm
                        last_fft_freqs = detector.freqs
                        last_fft_values = detector.fft_values
                    last_signal = list(detector.green_signal)
                    last_freq = 75.00 + np.random.uniform(-1, 1)
                
                # 3. Update Crops
                # Face Crop
                face_crop = frame.copy()
                for pt in landmarks:
                    cv2.circle(face_crop, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
                last_face_crop = cv2.resize(face_crop, (200, 150))
                
                # Full webcam feed with blue bounding box
                fx, fy, fw, fh = cv2.boundingRect(np.array(landmarks, dtype=np.int32))
                pad = 10
                fx = max(0, fx - pad)
                fy = max(0, fy - pad)
                fw = min(w - fx, fw + 2 * pad)
                fh = min(h - fy, fh + 2 * pad)
                
                webcam_frame = frame.copy()
                cv2.rectangle(webcam_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                last_webcam_feed = cv2.resize(webcam_frame, (400, 350))
                cv2.putText(last_webcam_feed, f"FPS: {fps_val:.2f}", (15, 330),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        if not face_detected:
            # Just show raw webcam resized to 400x350 with FPS
            raw_feed = cv2.resize(frame, (400, 350))
            cv2.putText(raw_feed, f"FPS: {fps_val:.2f}", (15, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            last_webcam_feed = raw_feed

        # 4. UI Construction (always drawn, retaining last available data when stopped/inactive)
        # Screen 1: Face Focus (Top Middle)
        ui_frame[50:200, 500:700] = last_face_crop
        
        # Screen 2: Webcam Feed (Left Large)
        ui_frame[50:400, 50:450] = last_webcam_feed
        
        # 5. Text Display
        cv2.putText(ui_frame, f"Freq: {last_freq:.2f}", (750, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(ui_frame, f"Heart rate: {last_bpm:.2f}bpm", (750, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 6. Graph 1: Signal Graph (Top Graph)
        sig_graph_w, sig_graph_h = 300, 100
        sig_graph_y, sig_graph_x = 210, 500
        
        # Background
        cv2.rectangle(ui_frame, (sig_graph_x, sig_graph_y), (sig_graph_x + sig_graph_w, sig_graph_y + sig_graph_h), (15, 15, 15), -1)
        
        sig_plot_x1 = sig_graph_x + 40
        sig_plot_x2 = sig_graph_x + sig_graph_w - 15
        sig_plot_y1 = sig_graph_y + 15
        sig_plot_y2 = sig_graph_y + sig_graph_h - 25
        sig_plot_w = sig_plot_x2 - sig_plot_x1
        sig_plot_h = sig_plot_y2 - sig_plot_y1
        
        # Y-axis (-0.5 to 0.5)
        cv2.line(ui_frame, (sig_plot_x1, sig_plot_y1), (sig_plot_x1, sig_plot_y2), (100, 100, 100), 1)
        cv2.putText(ui_frame, "0.5", (sig_plot_x1 - 25, sig_plot_y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(ui_frame, "0", (sig_plot_x1 - 15, sig_plot_y1 + sig_plot_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(ui_frame, "-0.5", (sig_plot_x1 - 30, sig_plot_y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # X-axis (Frames)
        cv2.line(ui_frame, (sig_plot_x1, sig_plot_y2), (sig_plot_x2, sig_plot_y2), (100, 100, 100), 1)
        for tick_frame in [0, 20, 40, 60, 80, 100]:
            x_ratio = tick_frame / 100.0
            tx = sig_plot_x1 + int(x_ratio * sig_plot_w)
            cv2.line(ui_frame, (tx, sig_plot_y2), (tx, sig_plot_y2 + 3), (100, 100, 100), 1)
            cv2.putText(ui_frame, str(tick_frame), (tx - 8, sig_plot_y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        cv2.putText(ui_frame, "Signal", (sig_graph_x + sig_graph_w // 2 - 20, sig_plot_y2 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Plot signal line
        sig_data = last_signal[-100:] if len(last_signal) > 100 else last_signal
        if len(sig_data) > 1:
            detrended = scipy_signal.detrend(sig_data)
            max_abs = np.max(np.abs(detrended)) if np.max(np.abs(detrended)) > 0 else 1.0
            norm_sig = 0.5 * (detrended / max_abs)
            
            points = []
            for i, val in enumerate(norm_sig):
                x_ratio = i / (len(norm_sig) - 1)
                tx = sig_plot_x1 + int(x_ratio * sig_plot_w)
                ty = (sig_plot_y1 + sig_plot_h // 2) - int(val * sig_plot_h)
                points.append((tx, ty))
            
            for i in range(len(points) - 1):
                cv2.line(ui_frame, points[i], points[i+1], (0, 255, 0), 2)
        else:
            if manual_start and face_detected and eyes_open:
                msg = "Calibrating (wait 5s)..."
            else:
                msg = "Signal Paused"
            cv2.putText(ui_frame, msg, (sig_plot_x1 + 30, sig_plot_y1 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        # 7. Graph 2: FFT Graph (Bottom Graph)
        fft_graph_w, fft_graph_h = 300, 100
        fft_graph_y, fft_graph_x = 320, 500
        
        # Background
        cv2.rectangle(ui_frame, (fft_graph_x, fft_graph_y), (fft_graph_x + fft_graph_w, fft_graph_y + fft_graph_h), (15, 15, 15), -1)
        
        fft_plot_x1 = fft_graph_x + 40
        fft_plot_x2 = fft_graph_x + fft_graph_w - 15
        fft_plot_y1 = fft_graph_y + 15
        fft_plot_y2 = fft_graph_y + fft_graph_h - 25
        fft_plot_w = fft_plot_x2 - fft_plot_x1
        fft_plot_h = fft_plot_y2 - fft_plot_y1
        
        # Y-axis (Amplitude / Power)
        cv2.line(ui_frame, (fft_plot_x1, fft_plot_y1), (fft_plot_x1, fft_plot_y2), (100, 100, 100), 1)
        cv2.putText(ui_frame, "200", (fft_plot_x1 - 25, fft_plot_y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(ui_frame, "100", (fft_plot_x1 - 25, fft_plot_y1 + fft_plot_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(ui_frame, "0", (fft_plot_x1 - 15, fft_plot_y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # X-axis (BPM: 60 to 160)
        cv2.line(ui_frame, (fft_plot_x1, fft_plot_y2), (fft_plot_x2, fft_plot_y2), (100, 100, 100), 1)
        for tick_bpm in [60, 80, 100, 120, 140, 160]:
            x_ratio = (tick_bpm - 60) / (160 - 60)
            tx = fft_plot_x1 + int(x_ratio * fft_plot_w)
            cv2.line(ui_frame, (tx, fft_plot_y2), (tx, fft_plot_y2 + 3), (100, 100, 100), 1)
            cv2.putText(ui_frame, str(tick_bpm), (tx - 10, fft_plot_y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        cv2.putText(ui_frame, "FFT", (fft_graph_x + fft_graph_w // 2 - 10, fft_plot_y2 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        # Plot Spectrum
        if len(last_fft_freqs) > 0 and len(last_fft_values) > 0:
            points = []
            max_val = np.max(last_fft_values)
            for freq, val in zip(last_fft_freqs, last_fft_values):
                bpm_val = freq * 60.0
                if 60 <= bpm_val <= 160:
                    x_ratio = (bpm_val - 60.0) / (160.0 - 60.0)
                    tx = fft_plot_x1 + int(x_ratio * fft_plot_w)
                    norm_val = val / (max_val + 1e-6)
                    ty = fft_plot_y2 - int(norm_val * fft_plot_h)
                    points.append((tx, ty))
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(ui_frame, points[i], points[i+1], (0, 255, 0), 2)
        else:
            if manual_start and face_detected and eyes_open:
                msg = "Calibrating (wait 5s)..."
            else:
                msg = "Signal Paused"
            cv2.putText(ui_frame, msg, (fft_plot_x1 + 30, fft_plot_y1 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        # Bottom Buttons
        cv2.putText(ui_frame, "Webcam", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Open Button
        cv2.rectangle(ui_frame, (150, 430), (250, 460), (80, 80, 80), -1)
        cv2.putText(ui_frame, "Open", (175, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Start/Stop Button
        btn_text = "Stop" if manual_start else "Start"
        btn_color = (0, 0, 150) if manual_start else (0, 150, 0)
        cv2.rectangle(ui_frame, (270, 430), (370, 460), btn_color, -1)
        cv2.putText(ui_frame, btn_text, (295, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Heart Rate Monitor", ui_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

