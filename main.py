import cv2
import mediapipe as mp
import math
import numpy as np

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
cmd_count = 0
color_red = (0,0,255)
color_blue = (255,0,0)
color_green = (0,255,0)

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

    # Assigning variables for Thumb and Index finger position
    if len(lmList) != 0:
      x1, y1 = lmList[4][1], lmList[4][2]
      x2, y2 = lmList[8][1], lmList[8][2]

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
    
    cv2.imshow('hand', image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cam.release()