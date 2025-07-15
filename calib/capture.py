import time, os
import picamera

# Output folder for calibration images
camera_number = os.popen("hostname").read().strip()[-1]
output_dir = os.path.expanduser(f"./pics/cam{camera_number}")
os.makedirs(output_dir, exist_ok=True)
os.system(f"rm -rf {output_dir}/*")

print("[INFO] Capturing calibration image set")
with picamera.PiCamera(resolution=(960, 640), framerate=20, sensor_mode=2) as camera:
    camera.start_preview(fullscreen=False, window=(100, 100, 1280, 960))
    time.sleep(5)
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = "off"
    camera.color_effects = (128, 128)
    g = camera.awb_gains
    camera.awb_mode = "off"
    camera.awb_gains = g

    num_images = 500
    for i in range(num_images):
        filename = f"{output_dir}/calib_{i:03d}.jpg"
        camera.capture(filename)
        print(f"Captured {filename}")
        time.sleep(0.05)

    camera.stop_preview()
