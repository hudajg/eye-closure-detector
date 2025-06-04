import cv2
import dlib
import pygame
import time
from scipy.spatial import distance
from imutils import face_utils

class SleepDetector:
    def __init__(self, predictor_path, alert_sound):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.alert_sound = alert_sound
        self.ear_thresh = 0.25
        self.frame_limit = 20
        self.counter = 0
        self.alert_triggered = False

        pygame.mixer.init()
        self.left_idx, self.right_idx = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"], face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def _play_alert(self):
        pygame.mixer.music.load(self.alert_sound)
        pygame.mixer.music.play()

    def _compute_ear(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def start(self):
        cam = cv2.VideoCapture(0)
        time.sleep(1.0)

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            for face in faces:
                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[self.left_idx[0]:self.left_idx[1]]
                right_eye = shape[self.right_idx[0]:self.right_idx[1]]

                # point add
                for (x, y) in left_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                for (x, y) in right_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                ear = (self._compute_ear(left_eye) + self._compute_ear(right_eye)) / 2.0

                if ear < self.ear_thresh:
                    self.counter += 1
                    if self.counter >= self.frame_limit and not self.alert_triggered:
                        cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self._play_alert()
                        self.alert_triggered = True
                else:
                    self.counter = 0
                    self.alert_triggered = False

                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Sleep Monitor", frame)
            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = SleepDetector("shape_predictor_68_face_landmarks.dat", "alert.wav")
    detector.start()