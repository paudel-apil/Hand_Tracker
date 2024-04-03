import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import time

win_name = 'Hand_Tracking'

cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    has_frame, frame = cap.read()
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y*h)
                print(id,cx,cy)
                if id == 9:
                    cv2.circle(frame,(cx,cy), 10, (255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    frame = cv2.flip(frame,1)
    cv2.putText(frame, str(int(fps)), (10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)
    cv2.imshow(win_name,frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyWindow(win_name)