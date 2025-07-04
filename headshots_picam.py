import cv2
from picamera2 import Picamera2
import os
import time

name = 'Jessica'  # Replace with your name

# Create dataset directory if it doesn't exist
save_dir = f"dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (512, 304)})
picam2.configure(config)
picam2.start()

# Allow camera to warm up
time.sleep(2)

img_counter = 0

while True:
    # Capture frame as numpy array
    frame = picam2.capture_array()
    
    # Show the image in a window
    cv2.imshow("Press Space to take a photo", frame)

    # Wait for key press
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = f"{save_dir}/image_{img_counter}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

cv2.destroyAllWindows()
