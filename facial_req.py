#! /usr/bin/python3

from picamera2 import Picamera2
import cv2
import face_recognition
import pickle
import time

# Load encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Initialize camera
print("[INFO] starting camera...")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})  # Lower res = faster
picam2.configure(config)
picam2.start()
time.sleep(1)

print("[INFO] running facial recognition...")
currentname = "unknown"

while True:
    frame = picam2.capture_array()

    # Resize frame to 1/2 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

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

    # Draw rectangles on the original (full-size) frame
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # Scale back up since the frame was downscaled
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 10 if top - 10 > 10 else top + 10
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Facial Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("[INFO] Exiting...")
        break

cv2.destroyAllWindows()
picam2.stop()
