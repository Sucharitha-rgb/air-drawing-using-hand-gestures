import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Drawing variables
prev_x, prev_y = 0, 0
draw_color = (0, 255, 0)
brush_thickness = 8

colors = [(0,255,0), (255,0,0), (0,0,255), (0,255,255)]
color_index = 0

def fingers_up(lm):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(lm[tips[0]].x < lm[tips[0]-1].x)

    # Other fingers
    for i in range(1,5):
        fingers.append(lm[tips[i]].y < lm[tips[i]-2].y)

    return fingers

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        h, w, _ = frame.shape

        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        finger_state = fingers_up(lm)
        total_fingers = finger_state.count(True)

        # DRAW (only index finger)
        if finger_state[1] and not finger_state[2]:
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
            prev_x, prev_y = x, y

        # MOVE (index + middle)
        elif finger_state[1] and finger_state[2]:
            prev_x, prev_y = 0, 0

        # ERASE (fist)
        elif total_fingers == 0:
            cv2.circle(canvas, (x, y), 40, (0,0,0), -1)

        # CLEAR (all fingers)
        elif total_fingers == 5:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # COLOR CHANGE (pinch)
        dist = np.hypot(
            lm[4].x - lm[8].x,
            lm[4].y - lm[8].y
        )
        if dist < 0.03:
            color_index = (color_index + 1) % len(colors)
            draw_color = colors[color_index]

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0

    # Merge canvas & frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.putText(frame, "AIR DRAWING - Press Q to Exit",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Air Drawing AR Tool", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


