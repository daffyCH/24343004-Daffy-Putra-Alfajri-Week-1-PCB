# ============================================
# LATIHAN 3: REAL-TIME VIDEO ENHANCEMENT
# ============================================

# --------------------------------------------
# Import Library
# --------------------------------------------
import cv2
import numpy as np
import time


# --------------------------------------------
# Class Real-time Enhancement
# --------------------------------------------
class RealTimeEnhancement:

    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []


    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame with real-time constraints

        Parameters:
        frame : Input video frame
        enhancement_type : Type of enhancement

        Returns:
        Enhanced frame
        """

        # Convert ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------------
        # Enhancement Method
        # ---------------------------------
        if enhancement_type == 'adaptive':

            # CLAHE (adaptive histogram equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

        elif enhancement_type == 'histogram':

            enhanced = cv2.equalizeHist(gray)

        elif enhancement_type == 'denoise':

            enhanced = cv2.GaussianBlur(gray, (5,5), 0)

        else:
            enhanced = gray


        # ---------------------------------
        # Temporal Consistency
        # ---------------------------------
        self.history_buffer.append(enhanced)

        if len(self.history_buffer) > 5:
            self.history_buffer.pop(0)

        enhanced = np.mean(self.history_buffer, axis=0).astype(np.uint8)

        return enhanced


# --------------------------------------------
# Main Program
# --------------------------------------------
def main():

    print("=== LATIHAN 3: REAL-TIME VIDEO ENHANCEMENT ===")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: kamera tidak dapat diakses")
        return

    enhancer = RealTimeEnhancement(target_fps=30)

    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Enhancement frame
        enhanced_frame = enhancer.enhance_frame(frame, 'adaptive')

        # Hitung FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Tampilkan FPS
        cv2.putText(enhanced_frame,
                    f"FPS: {int(fps)}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255),
                    2)

        # Tampilkan frame
        cv2.imshow("Original Video", frame)
        cv2.imshow("Enhanced Video", enhanced_frame)

        # tekan q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------
# Run Program
# --------------------------------------------
if __name__ == "__main__":
    main()