import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, draw=True, display=True):

    output_image = image.copy()
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)

    if results.multi_hand_landmarks and draw:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(image = output_image, landmark_list = hand_landmarks, connections = mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec = mp_drawing.DrawingSpec(color = (255,255,255), thickness = 2, circle_radius = 2),
                                      connection_drawing_spec = mp_drawing.DrawingSpec(color = (0,255,0), thickness = 2, circle_radius = 2))

    if display:

        plt.figure(figsize = [15,15])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axix('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axix('off');
        
    else:

        return output_image, results


#image = cv2.imread()
#detectHandsLandmarks(image, hands, display=True)

def countfingers(image, results, draw=True, display=True):
    
    height,width,_ = image.shape
    output_image = image.copy()
    count = {'RIGHT': 0, 'LEFT': 0}
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    for hand_index, hand_info in enumerate(results.multi_handedness):

        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        for tip_index in fingers_tips_ids:
            
            finger_name = tip_index.name.split('_')[0]

            if(hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):

                fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                count[hand_label.upper()] += 1

        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        if (hand_label=='RIGHT' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='LEFT' and (thumb_tip_x > thumb_mcp_x)):

            fingers_statuses[hand_label.upper()+"_THUMB"] = True
            count[hand_label.upper()] += 1

    if draw:

        cv2.putText(output_image, "Total Fingers: ",(10, 25), cv2.FONT_HERSHEY_COMPLEX, 1,(20,255,155), 2)
        cv2.putText(output_image, str(sum(count.values())), (width//2-150,240), cv2.FONT_HERSHEY_SIMPLEX, 8.9, (20,25,155), 10,10)

    if display:

        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axix('off');

    else:

        return output_image, fingers_statuses, count


camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

cv2.namedWindow("Finger Counter", cv2.WINDOW_NORMAL)

while camera_video.isOpened():

    ok, frame = camera_video.read()

    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    frame, results = detectHandsLandmarks(frame, hands_videos, display=False)

    if results.multi_hand_landmarks:

        frame, fingers_statuses, count = countfingers(frame, results, display=False)

    cv2.imshow("Fingers Conter", frame)
    k = cv2.waitKey(1) & 0xFF

    if(k==(27)):
       break

camera_video.release()
cv2.destroyAllWindows()
