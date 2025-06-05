import os, cv2, time, socket, argparse, logging, traceback
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# -------------------------------
# Command-line arguments
# -------------------------------
parser = argparse.ArgumentParser(description="Blob tracker for MoCap system")
parser.add_argument("-high", type=int, default=210, help="High threshold for blobs")
parser.add_argument("-area", type=float, default=8.0, help="Minimum area for blobs")
parser.add_argument(
    "--kernel", type=int, default=0, help="Morph close kernel size (0 = off)"
)
parser.add_argument("--host", type=str, default="nuc.local", help="Server hostname/IP")
parser.add_argument("--port", type=int, default=8888, help="UDP port")
parser.add_argument("--display", action="store_true", help="Display result frames")
parser.add_argument("--save", action="store_true", help="Save debug output images")
args = parser.parse_args()

# -------------------------------
# Setup
# -------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
os.system("rm -rf /dev/shm/*.bmp")
pid = os.getpid()
os.system(f"sudo renice -n -19 -p {pid}")
times, frames = [], []
hostnamePC = socket.gethostbyname(args.host)
UDPSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------------
# Blob Detector
# -------------------------------
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = args.high
params.maxThreshold = 255
params.thresholdStep = 5
params.minDistBetweenBlobs = 0
params.filterByColor = True
params.blobColor = 255
params.filterByArea = True
params.minArea = args.area
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
detector = cv2.SimpleBlobDetector_create(params)


# -------------------------------
# Image processing coroutine
# -------------------------------
def imageProcessing():
    counter = 0
    bitsShift, constMultiplier = 4, 16
    log.info("Starting image processing loop...")
    while True:
        try:
            log.debug("Waiting for new image...")
            start = time.time()
            img, ts = yield
            if img is None or img.size == 0:
                log.debug("[DEBUG] Empty image, skipping...")
                continue

            _, thresh = cv2.threshold(img, args.high, 255, cv2.THRESH_BINARY)

            if args.kernel > 0:
                kernel = np.ones((args.kernel, args.kernel), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            crop_flag = False
            if crop_flag:
                coord = cv2.findNonZero(thresh)
                if coord is None or coord.size == 0:
                    log.debug("[DEBUG] No bright regions found.")
                    continue

                coord = coord.reshape(-1, 2).T
                xMin, xMax = int(min(coord[1])), int(max(coord[1]))
                yMin, yMax = int(min(coord[0])), int(max(coord[0]))
                x1, x2 = max(0, xMin - 5), min(img.shape[0], xMax + 5)
                y1, y2 = max(0, yMin - 5), min(img.shape[1], yMax + 5)

                cropped = img[x1:x2, y1:y2]
                if cropped.size == 0:
                    log.debug("[DEBUG] Cropped region empty, skipping.")
                    continue
            else:
                xMin, xMax = 0, img.shape[0]
                yMin, yMax = 0, img.shape[1]
                cropped = img

            keypoints = detector.detect(cropped)
            N = len(keypoints)
            msg = np.zeros(N * 3 + 4)
            for i, keypt in enumerate(keypoints):
                x, y, size = keypt.pt[0], keypt.pt[1], keypt.size
                msg[(i << 1) + i] = x
                msg[(i << 1) + i + 1] = y
                msg[(i << 1) + i + 2] = size
                log.debug(f"[BLOB {i}] x: {x:.2f}, y: {y:.2f}, size: {size:.2f}")
            msg[-4], msg[-3], msg[-2], msg[-1] = xMin, yMin, ts, counter

            log.info(f"Sending {N} blobs...")
            UDPSocket.sendto(msg.tobytes(), (hostnamePC, args.port))

            # Visualization
            x1, x2 = max(0, xMin - 10), min(img.shape[0], xMax + 10)
            y1, y2 = max(0, yMin - 10), min(img.shape[1], yMax + 10)
            region = img[x1:x2, y1:y2]
            imgWithKPts = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)

            for keyPt in keypoints:
                center = (
                    int(np.round(keyPt.pt[0] * constMultiplier)),
                    int(np.round(keyPt.pt[1] * constMultiplier)),
                )
                radius = int(np.round(keyPt.size / 2 * constMultiplier))
                cv2.circle(
                    imgWithKPts,
                    center,
                    radius,
                    (255, 0, 0),
                    1,
                    lineType=16,
                    shift=bitsShift,
                )
                cv2.circle(imgWithKPts, center, 1, (0, 0, 255), -1, shift=bitsShift)

            if args.display:
                cv2.imshow("Blobs", imgWithKPts)
                cv2.waitKey(10)

            if args.save:
                cv2.imwrite(f"/dev/shm/debug_frame_{ts}.png", imgWithKPts)

            frames.append(imgWithKPts)
            times.append(time.time() - start)
            counter += 1

        except GeneratorExit:
            log.info("Image processing coroutine closed.")
            return
        except Exception:
            log.error("Error during image processing:", exc_info=True)


# -------------------------------
# Watchdog filesystem handler
# -------------------------------
class Handler(FileSystemEventHandler):
    counter = 0
    coRout = imageProcessing()
    coRout.__next__()
    lastImg = ""

    @staticmethod
    def on_any_event(event):
        if event.is_directory or event.event_type != "created":
            return
        if Handler.counter:
            name = Handler.lastImg
            path = f"/dev/shm/{name}.bmp"
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                Handler.coRout.send((img, int(name)))
            os.remove(path)
        Handler.lastImg = event.src_path[-14:-4]
        Handler.counter += 1


# -------------------------------
# Directory watch wrapper
# -------------------------------
class OnMyWatch:
    watchDirectory = "/dev/shm/"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        self.observer.schedule(Handler(), self.watchDirectory, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(300)
                UDPSocket.sendto(np.array([0.0]).tobytes(), (hostnamePC, args.port))
                self.observer.stop()
                log.info("Observer Stopped")
                break
        except:
            UDPSocket.sendto(np.array([0.0]).tobytes(), (hostnamePC, args.port))
            self.observer.stop()
            log.warning("Observer Interrupted")
        self.observer.join()


# -------------------------------
# Main loop
# -------------------------------
if __name__ == "__main__":
    watch = OnMyWatch()
    watch.run()
    if times:
        times = np.array(times[1:])
        log.info(f"[RESULTS] Processing at {round(1 / np.mean(times), 2)} FPS")
        log.info(f"[RESULTS] {len(times)} valid images")
    else:
        log.warning("[RESULTS] No valid images captured")

    log.info("Display frames with OpenCV...")
    for fid, frame in enumerate(frames):
        log.info(f" Displaying Frame {fid}")
        cv2.imshow("Blobs", frame)
        cv2.waitKey(10)

    cv2.destroyAllWindows()
