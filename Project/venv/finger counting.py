import cv2
import mediapipe as mp

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])
            
            if landmark_list:
                fingers = []
                
                # Thumb
                if landmark_list[4][0] < landmark_list[3][0]:  # Right hand
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # 4 Fingers
                for tip_id in [8, 12, 16, 20]:
                    if landmark_list[tip_id][1] < landmark_list[tip_id - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                total_fingers = fingers.count(1)
                
                cv2.putText(frame, str(total_fingers), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
