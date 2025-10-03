#!/usr/bin/python3

from picamera2 import Picamera2
import cv2
import face_recognition
import pickle
import time

from gpiozero import Motor

from time import sleep
import numpy as np
import tflite_runtime.interpreter as tflite

# --- Servo Setup ---
pos = ""  # None = unknown, 'left', 'right', 'center'

left_motor = Motor(forward=22, backward=17)
right_motor = Motor(forward=5, backward=27)

right_motor_inverted = True

MOVE_COOLDOWN = 2  # seconds
last_move_time = 0 

def servo_control(direction, speed=1):
    global pos
    if direction == pos:
        return  # already in position

    print(f"Moving {direction} at speed {speed}")
    if direction == "right":
        left_motor.forward(speed)
        if right_motor_inverted:
            right_motor.backward(speed)
        else:
            right_motor.forward(speed)
    elif direction == "left":
        left_motor.backward(speed)
        if right_motor_inverted:
            right_motor.forward(speed)
        else:
            right_motor.backward(speed)
    elif direction == "center":
        left_motor.stop()
        right_motor.stop()
    else:
        print("Invalid direction")
        return

    sleep(0.5)
    left_motor.stop()
    right_motor.stop()
    pos = direction

# --- Load Facial Encodings ---
print("[INFO] Loading facial encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# --- Load MoveNet Model ---
interpreter = tflite.Interpreter(model_path="movenet_lightning.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def detect_pose_tflite(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(frame_rgb, (192, 192))

    if input_details[0]['dtype'] == np.uint8:
        input_data = np.expand_dims(input_image.astype(np.uint8), axis=0)
    else:
        input_data = np.expand_dims(input_image.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints

# --- Initialize Camera ---
print("[INFO] Starting camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()
time.sleep(1)

currentname = "unknown"
print("[INFO] Running facial and pose recognition...")

POSE_CONNECTIONS = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    keypoint_coords = {}
    landmarks = detect_pose_tflite(frame)

    if landmarks is not None and landmarks.size > 0:
        keypoints = landmarks[0][0]

        for i, (y, x, score) in enumerate(keypoints):
            if score > 0.3:
                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])
                keypoint_coords[i] = (px, py)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        for pt1, pt2 in POSE_CONNECTIONS:
            if pt1 in keypoint_coords and pt2 in keypoint_coords:
                cv2.line(frame, keypoint_coords[pt1], keypoint_coords[pt2], (255, 0, 0), 2)

        # Calculate mid-hip x pos if hips detected
        if 11 in keypoint_coords and 12 in keypoint_coords:
            mid_hip_x = (keypoint_coords[11][0] + keypoint_coords[12][0]) // 2
        else:
            mid_hip_x = frame.shape[1] // 2  # default center

        frame_center_x = frame.shape[1] // 2
        offset_x = mid_hip_x - frame_center_x
 
        now = time.time()
        if now - last_move_time > MOVE_COOLDOWN:
            if offset_x < 30:
                servo_control("right")
                last_move_time = now
            elif offset_x > -30:
                servo_control("left")
                last_move_time = now
            else:
                servo_control("center")
                last_move_time = now

    # --- Face Recognition ---
    boxes = face_recognition.face_locations(rgb_small_frame)
    encodings = face_recognition.face_encodings(rgb_small_frame, boxes)
    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
            if currentname != name:
                currentname = name
                print(f"[INFO] Recognized: {currentname}")
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        cv2.putText(frame, name, (left, top - 10 if top > 10 else top + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        face_center_x = (left + right) // 2
        face_center_y = (top + bottom) // 2
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        offset_x_face = face_center_x - frame_center_x
        offset_y_face = face_center_y - frame_center_y

        cv2.putText(frame, f"Offset: ({offset_x_face},{offset_y_face})", (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if -50 < offset_x_face < 50 and -50 < offset_y_face < 50:
            cv2.putText(frame, "CENTERED!", (left, bottom + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- Show Frame ---
    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

# --- Cleanup ---
cv2.destroyAllWindows()
picam2.stop()
