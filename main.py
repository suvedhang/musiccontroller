import cv2
import mediapipe as mp
import pyautogui
import time
# import math # Math was not used, so I removed it

# --- Initialization ---
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Cooldown variables to prevent multiple actions for a single gesture
cooldown_period = 2  # 2 seconds cooldown
last_action_time = 0
last_action = "None"


# --- Helper Function ---
# --- MODIFIED to accept hand_label ---
def get_finger_status(hand_landmarks, hand_label="Right"):
    """
    Returns a dictionary indicating whether each finger is up or down.
    Takes hand_label ('Left' or 'Right') to correctly process the thumb.
    """
    finger_tips = {
        'THUMB': 4, 'INDEX': 8, 'MIDDLE': 12, 'RING': 16, 'PINKY': 20
    }
    status = {}

    thumb_tip_x = hand_landmarks.landmark[finger_tips['THUMB']].x
    thumb_mcp_x = hand_landmarks.landmark[finger_tips['THUMB'] - 2].x

    # --- MODIFIED THUMB LOGIC ---
    # This logic is now robust for either hand because the camera is flipped.
    # For a RIGHT hand, an open thumb has a tip.x that is LESS than its base.x
    if hand_label == "Right":
        status['THUMB'] = thumb_tip_x < thumb_mcp_x
    # For a LEFT hand, an open thumb has a tip.x that is GREATER than its base.x
    else:  # hand_label == "Left"
        status['THUMB'] = thumb_tip_x > thumb_mcp_x
    # --- END OF MODIFIED LOGIC ---

    # For other fingers, we check the y-coordinate (this logic was correct)
    for finger, tip_index in finger_tips.items():
        if finger == 'THUMB':
            continue

        finger_tip_y = hand_landmarks.landmark[tip_index].y
        finger_pip_y = hand_landmarks.landmark[tip_index - 2].y
        status[finger] = finger_tip_y < finger_pip_y

    return status

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a natural selfie-view
    img = cv2.flip(img, 1)

    # Convert the BGR image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(img_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        # --- NEW: Get Handedness (Left or Right) ---
        hand_label = "Unknown"
        if results.multi_handedness:
            # Get the classification (label) for the first hand detected
            classification = results.multi_handedness[0].classification[0]
            hand_label = classification.label  # Will be 'Left' or 'Right'
        # --- END OF NEW CODE ---

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check for gestures only if cooldown has passed
        if current_time - last_action_time > cooldown_period:
            
            # --- MODIFIED: Pass hand_label to the function ---
            finger_status = get_finger_status(hand_landmarks, hand_label)

            # GESTURE 1: Play/Pause (Open Palm - all 5 fingers up)
            if all(finger_status.values()):
                pyautogui.press('space')
                last_action = "Play/Pause"
                last_action_time = current_time

            # GESTURE 2: Fast Forward (Index and Pinky up)
            elif finger_status['INDEX'] and finger_status['PINKY'] and not finger_status['MIDDLE'] and not finger_status['RING']:
                pyautogui.press('right')
                last_action = "Fast Forward 5s"
                last_action_time = current_time

            # GESTURE 3: Rewind (Index and Middle up - "Peace" sign)
            elif finger_status['INDEX'] and finger_status['MIDDLE'] and not finger_status['RING'] and not finger_status['PINKY']:
                pyautogui.press('left')
                last_action = "Rewind 5s"
                last_action_time = current_time

    # --- Display Information ---
    if current_time - last_action_time < cooldown_period:
        cv2.putText(img, f"Action: {last_action} (Cooldown)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(img, f"Ready for gesture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Controlled Media Player", img)

    # Exit condition: Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()