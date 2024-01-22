import cv2
import mediapipe as mp
import math
import numpy as np
import os
import pyautogui as pg

class HandGestureController:
    def __init__(self):
        self.output_folder = 'images'
        os.makedirs(self.output_folder, exist_ok=True)

        # mediapipe initialize
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # Webcam initialize
        self.wCam, self.hCam = 640, 480
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, self.wCam)
        self.cam.set(4, self.hCam)

        # Variables initialize
        self.tolerance = 10
        self.sensitivity = 2
        self.fist_threshold = 50
        self.circle_threshold = 120
        self.image_count = 0
        self.color_red = (0, 0, 255)
        self.color_blue = (255, 0, 0)
        self.color_green = (0, 255, 0)
        self.choose_color = "negative"

    def process_hand(self):
        with self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while self.cam.isOpened():
                success, image = self.cam.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.detect_hand(results, image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                #TODO check this
            self.cam.release()
        cv2.destroyAllWindows()        
            

        

    def detect_hand(self, results, image):
        fingerCount = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                handLandmarks = []
                for landmark in hand_landmarks.landmark:
                    handLandmarks.append([landmark.x, landmark.y])
                

                # Detect witch hand is raised 
                # Test conditions for fingers we increase the count if the finger raised.  
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount = fingerCount+1
                    cv2.putText(image,"Right raised", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
          
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount = fingerCount+1
                    cv2.putText(image,"Left raised", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
 

                # Other fingers: tip y position must be lower than pip y position, 
                # as image origin is in the upper left corner. good damn he right
                    
                if handLandmarks[8][1] < handLandmarks[6][1]:#Index finger
                    fingerCount = fingerCount+1
                if handLandmarks[12][1] < handLandmarks[10][1]:#Middle finger
                    fingerCount = fingerCount+1
                if handLandmarks[16][1] < handLandmarks[14][1]:#Ring finger
                    fingerCount = fingerCount+1
                if handLandmarks[20][1] < handLandmarks[18][1]:#little finger
                    fingerCount = fingerCount+1

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                self.process_fingers(handLandmarks, image)

            # Dispaly The count Fingers
            cv2.putText(image, f'Finger up={str(fingerCount)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if self.choose_color == "gray":
                # Convert the image to gray
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Display the gray image
                cv2.imshow('Hands (Gray)', gray_image)
            elif self.choose_color == "negative":
                #Convert the image to negative
                negative_image = abs(255 - image)
                #Display the negative image
                cv2.imshow('Hands (Negative)', negative_image)
            else:
                #Display the RGB image
                cv2.imshow('hand', image)

    def process_fingers(self, handLandmarks, image):

        #Assigning variables for fingers position
        x1, y1 = int(handLandmarks[4][0] * self.wCam), int(handLandmarks[4][1] * self.hCam)  # Thumb finger
        x2, y2 = int(handLandmarks[8][0] * self.wCam), int(handLandmarks[8][1] * self.hCam)  # Index finger
        x3, y3 = int(handLandmarks[12][0] * self.wCam), int(handLandmarks[12][1] * self.hCam)  # Middle finger
        x4, y4 = int(handLandmarks[16][0] * self.wCam), int(handLandmarks[16][1] * self.hCam)  # Ring finger
        x5, y5 = int(handLandmarks[20][0] * self.wCam), int(handLandmarks[20][1] * self.hCam)  # Little finger

        #calcul the distance between the fingers
        thumb_index_dist = math.hypot(x2 - x1, y2 - y1)
        index_middle_dist = math.hypot(x3 - x2, y3 - y2)
        middle_ring_dist = math.hypot(x4 - x3, y4 - y3)
        ring_little_dist = math.hypot(x5 - x4, y5 - y4)

        #calcul the angels for precessing the detection of circles
        thumb_index_angle = math.degrees(math.atan2(y1 - y2, x1 - x2))

        #Control the mouse 
        if index_middle_dist > self.fist_threshold and handLandmarks[20][1] > handLandmarks[18][1] and handLandmarks[16][1] > handLandmarks[14][1]:
            pg.moveTo((x2 + x3) / 2 * self.sensitivity, (y2 + y3) / 2 * self.sensitivity)
            #TODO we need to improve the precision of this controlle
            
        #Perform a click
        elif handLandmarks[8][1] < handLandmarks[6][1] and handLandmarks[12][1] < handLandmarks[10][1] and handLandmarks[16][1] > handLandmarks[14][1] and handLandmarks[20][1] > handLandmarks[18][1]:
            pg.click()  # Perform a mouse click

        #close hand completely to save the image in RGB format
        if all(dist < self.fist_threshold for dist in [thumb_index_dist, index_middle_dist, middle_ring_dist, ring_little_dist]):
            self.save_image(image, 'RGB')
        #close hand and let only the little finger open to save it in GRAY format    
        elif all(dist < self.fist_threshold for dist in [thumb_index_dist, index_middle_dist, middle_ring_dist]):
            self.save_image(image, 'gray')

        #Display circle
        if thumb_index_dist < self.fist_threshold and abs(thumb_index_angle) < self.circle_threshold:
            cv2.putText(image, 'Circle', (self.wCam - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_green, 2)
            

    #For the colors the text where is displayed dont take care NOT IMPORTANT !!!!!!!
    def save_image(self, image, image_type):
        if image_type == 'gray':
            cv2.putText(image, 'Frame saved on gray format', (10, self.hCam - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        self.color_blue, 2)
            self.image_count += 1
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images_filename = os.path.join(self.output_folder, f'image_gray_{self.image_count}.png')
            cv2.imwrite(images_filename, gray_img)
        elif image_type == 'RGB':
            cv2.putText(image, 'Frame saved on RGB format', (10, self.hCam - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        self.color_red, 2)
            self.image_count += 1
            images_filename = os.path.join(self.output_folder, f'image_{self.image_count}.png')
            cv2.imwrite(images_filename, image)