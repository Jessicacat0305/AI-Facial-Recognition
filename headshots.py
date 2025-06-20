from picamera2 import Picamera2, Preview
import cv2
import os

name = 'Lucas'
save_dir = os.path.join("dataset", name)
os.makedirs(save_dir, exist_ok=True)

# Initialize the Picamera2 object
picam2 = Picamera2()

# Configure the camera preview and set format and resolution
config = picam2.create_still_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)

# Start the camera preview
picam2.start()

cv2.namedWindow("Press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Press space to take a photo", 500, 300)

img_counter = 0

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Show the frame
    cv2.imshow("Press space to take a photo", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed, save the image
        img_name = os.path.join(save_dir, f"image_{img_counter}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

# Clean up and release the camera
cv2.destroyAllWindows()
picam2.stop()
