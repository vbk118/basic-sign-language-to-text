import cv2
import numpy as np

# Define HSV range for skin color (can adjust for lighting/skin tone)
lower_skin = np.array([0, 30, 60], dtype=np.uint8)
upper_skin = np.array([20, 150, 255], dtype=np.uint8)

# Gesture mapping (very basic placeholder)
gesture_map = {
    "OPEN_HAND": "Hello",
    "FIST": "Fist"
}

def detect_gesture(contour):
    area = cv2.contourArea(contour)
    if area > 40000:  # big contour → open hand
        return "OPEN_HAND"
    elif area > 10000:  # smaller → fist
        return "FIST"
    else:
        return None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold to get skin region
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        gesture = detect_gesture(max_contour)
        if gesture:
            text = gesture_map.get(gesture, "")
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3, cv2.LINE_AA)

        # Draw contour
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 3)

    cv2.imshow("Hand Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
