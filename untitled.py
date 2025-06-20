from picamera2 import Picamera2
import face_recognition
import cv2
import pickle
import time

# Load encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Check encodings
print(f"[INFO] Loaded {len(data['encodings'])} encodings.")
if len(data["encodings"]) == 0:
    print("[ERROR] No encodings found.")

# Initialize PiCamera2
print("[INFO] Initializing Pi Camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})  # Increased resolution
picam2.configure(config)
picam2.start()

# Give camera time to warm up
time.sleep(5)

print("[INFO] Capturing frame...")
frame = picam2.capture_array()
rgb = frame[:, :, ::-1]  # Convert BGR to RGB

# Display captured frame for debugging
cv2.imshow("Captured Frame", frame)
cv2.waitKey(1)  # Wait for 1 ms

# Detect face locations and encodings
print("[INFO] Detecting faces...")
boxes = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, boxes)

print(f"Detected face locations: {boxes}")
if not boxes:
    print("[INFO] No faces detected!")
else:
    print(f"[INFO] Detected {len(boxes)} face(s).")

# Draw results
for (top, right, bottom, left), encoding in zip(boxes, encodings):
    name = "Unknown"
    
    matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(data["encodings"], encoding)

    if matches:
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = data["names"][best_match_index]

    print(f"[INFO] Match: {name}")

    # Draw box and label
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, name, (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Show result
cv2.imshow("Debug Live Face Recognition", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
picam2.stop()
