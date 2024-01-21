import cv2
import mediapipe as mp
import math
import numpy as np
import os
import pyautogui as pg
import time

output_folder = 'images'
os.makedirs(output_folder, exist_ok=True)

# mediapipe initialize
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam initialize
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)

# Variables initialize
cmd_mode = False
seuil = 10
sensitivity = 2
fist_threshold = 50
cmd_count = 0
image_count = 0
color_red = (0,0,255)
color_blue = (255,0,0)
color_green = (0,255,0)


#Exemple of using control mouse #TODO I will use it later with the hand to control



with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:



  while cam.isOpened():

    # capture the image
    success, image = cam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # display commande mode state
    if cmd_mode:
      cv2.circle(image, (50, 50), 15, color_green, 3)
    else:
      cv2.circle(image, (50, 50), 15, color_red, 3)

    # draw the hand annotations on the image.
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
            )

    # find postion of Hand landmarks      
    lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])          

    # Assigning variables for fingers position
    if len(lmList) != 0:
      x1, y1 = lmList[4][1], lmList[4][2] # Thumb finger
      x2, y2 = lmList[8][1], lmList[8][2] #Index finger
      x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger
      x4, y4 = lmList[16][1], lmList[16][2]  # Ring finger
      x5, y5 = lmList[20][1], lmList[20][2]  # Little finger


      # Calculate angels between fingers 
      #Try change between y2 and y1 same for x1 and x2 and u will see
      thumb_index_angle = math.degrees(math.atan2(y1 - y2, x1 - x2)) 

      # Calculate distances between fingers
      thumb_index_dist = math.hypot(x2 - x1, y2 - y1)
      index_middle_dist = math.hypot(x3 - x2, y3 - y2)
      middle_ring_dist = math.hypot(x4 - x3, y4 - y3)
      ring_little_dist = math.hypot(x5 - x4, y5 - y4)

      # Marking Thumb and Index finger
      cv2.circle(image, (x1,y1),15,(255,255,255))  
      cv2.circle(image, (x2,y2),15,(255,255,255))  

      cv2.line(image,(x1,y1),(x2,y2),color_blue,2)

      length = math.hypot(x2-x1,y2-y1)
      if length > 200:
        cv2.line(image,(x1,y1),(x2,y2),color_green,2)
        
        if cmd_count <= seuil: 
            cmd_count += 1

      elif length < 100:
        cv2.line(image,(x1,y1),(x2,y2),color_red,2)
        
        if cmd_count >= 0:
          cmd_count-= 1
    
      if cmd_count >= seuil:
        cmd_mode = True
      elif cmd_count <= 0:
        cmd_mode = False

      # Move the mouse based on hand position
      if length > 50:
        pg.moveTo(x1 * sensitivity, y1 * sensitivity)

      #Save frames and save it as gray image Only the window of the video  (close all your fingers)
      elif all(dist < fist_threshold for dist in [thumb_index_dist, index_middle_dist, middle_ring_dist, ring_little_dist]):
        cv2.putText(image, 'Closed Fist', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_red, 2)
        image_count += 1
        images_filename = os.path.join(output_folder, f'image_{image_count}.png')
        cv2.imwrite(images_filename, image)

      #Take a screenshot of the whole screen of the PC (Close all fingers only the little one)
      elif all(dist < fist_threshold for dist in [thumb_index_dist, index_middle_dist, middle_ring_dist]):
        cv2.putText(image, 'little dist used',(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color_blue,2)
        im2 = pg.screenshot('my_screenshot.png')
      
      #Navigate in web (close all the fingers only ring finger) 
      elif all(dist < fist_threshold for dist in [thumb_index_dist, index_middle_dist,ring_little_dist]):
        pg.moveTo(-833,-152, duration=1)
        pg.click(-833,-152, duration=1)
        pg.moveTo(-81645,-118, duration=1)
        pg.write('youtube')
        pg.press('enter')
        pg.moveTo(-1666,150, duration=1)

      elif thumb_index_dist < fist_threshold and abs(thumb_index_angle) < 60:
        cv2.putText(image, 'Circle', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2)
        #Do Something here


    cv2.imshow('hand', image) 

      
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break


cam.release()
cv2.destroyAllWindows()