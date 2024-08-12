import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#IMPORT DEPENDENCIES
import cv2
import matplotlib.pyplot as plt
from Function import HandFaceDetector
from Config import mp_holistic
obj = HandFaceDetector()

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = obj.mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        obj.draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

obj.draw_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))