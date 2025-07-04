from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500
import cv2, os

# Setup
name = 'Jessica'
save_dir = os.path.join("dataset", name)
os.makedirs(save_dir, exist_ok=True)

# Initialize Picamera2 and attach IMX500 model
pose_model = "/usr/share/imx500-models/imx500_network_posenet_higherhrnet_pp.rpk"
imx500 = IMX500(pose_model)
picam2 = Picamera2()
picam2.register_device(imx500)

# Configure preview stream
config = picam2.create_preview_configuration(main={"format":"RGB888","size":(640,480)})
picam2.configure(config)
picam2.start()

# Window for saving frames
cv2.namedWindow("AI Pose Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Pose Tracking", 640, 480)
img_counter = 0

while True:
    arr = picam2.capture_array()
    # Tensor is in metadata, not overlayed
    request = picam2.capture_request()
    tensor = imx500.get_outputs(request.request)
    # Now you must visualize it yourself (e.g. using IMX500.convert_inference_coords + plot)
    frame = arr  # placeholder
    cv2.imshow("AI Pose Tracking", frame)
    key = cv2.waitKey(1)
    if key == 27: break
    if key == 32:
        filename = os.path.join(save_dir, f"pose_{img_counter}.jpg")
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        img_counter += 1

cv2.destroyAllWindows()
picam2.stop()
