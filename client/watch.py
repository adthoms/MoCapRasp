from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import cv2, os, socket, time, argparse
import traceback

# -------------------------------
# Command-line argument parsing
# -------------------------------
parser = argparse.ArgumentParser(
    description="""Image processing client for the MoCap system at the Erobotica lab of UFCG.
Use it with the corresponding server script.""",
    add_help=False,
)
parser.add_argument("-high", type=int, default=240, help="High threshold for bright blobs")
parser.add_argument("-area", type=float, default=2.0, help="Minimum area for blobs")
parser.add_argument("--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
args = parser.parse_args()

os.system("rm -rf /dev/shm/*.bmp")

# -------------------------------
# Blob detector setup
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
params.filterByConvexity = False
params.filterByCircularity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

# -------------------------------
# Prioritize CPU scheduling
# -------------------------------
pid = os.getpid()
os.system(f"sudo renice -n -19 -p {pid}")
times = []
frames = []

# -------------------------------
# UDP setup for sending data
# -------------------------------
UDPSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
hostnamePC = socket.gethostbyname("nuc.local")

# -------------------------------
# Image processing coroutine
# -------------------------------
def imageProcessing():
    counter = 0
    bitsShift, constMultiplier = 4, 16
    print("Starting image processing loop...")
    while True:
        try:
            print("Waiting for new image...")
            start = time.time()
            img, ts = yield
            if img is None or img.size == 0:
                print("[DEBUG] Empty image, skipping...")
                continue

            _, thresh = cv2.threshold(img, args.high, 255, cv2.THRESH_BINARY)
            coord = cv2.findNonZero(thresh)
            if coord is None or coord.size == 0:
                print("[DEBUG] No bright regions found.")
                continue

            coord = coord.reshape(-1, 2).T
            xMin, xMax = int(min(coord[1])), int(max(coord[1]))
            yMin, yMax = int(min(coord[0])), int(max(coord[0]))
            x1, x2 = max(0, xMin - 5), min(img.shape[0], xMax + 5)
            y1, y2 = max(0, yMin - 5), min(img.shape[1], yMax + 5)

            cropped = img[x1:x2, y1:y2]
            if cropped.size == 0:
                print("[DEBUG] Cropped region empty, skipping.")
                continue

            keypoints = detector.detect(cropped)
            N = len(keypoints)
            msg = np.zeros(N * 3 + 4)
            for i in range(N):
                msg[(i << 1) + i], msg[(i << 1) + i + 1], msg[(i << 1) + i + 2] = (
                    keypoints[i].pt[0], keypoints[i].pt[1], keypoints[i].size
                )
            msg[-4], msg[-3], msg[-2], msg[-1] = xMin, yMin, ts, counter

            print(f"[INFO] Sending {N} blobs...")
            UDPSocket.sendto(msg.tobytes(), (hostnamePC, 8888))

            # Visualization (optional)
            x1, x2 = max(0, xMin - 10), min(img.shape[0], xMax + 10)
            y1, y2 = max(0, yMin - 10), min(img.shape[1], yMax + 10)
            cropped_region = img[x1:x2, y1:y2]
            imgWithKPts = cv2.cvtColor(cropped_region, cv2.COLOR_GRAY2BGR)

            for keyPt in keypoints:
                center = (
                    int(np.round(keyPt.pt[0] * constMultiplier)),
                    int(np.round(keyPt.pt[1] * constMultiplier)),
                )
                radius = int(np.round(keyPt.size / 2 * constMultiplier))
                cv2.circle(imgWithKPts, center, radius, (255, 0, 0), 1, lineType=16, shift=bitsShift)
                cv2.circle(imgWithKPts, center, 1, (0, 0, 255), -1, shift=bitsShift)

            frames.append(imgWithKPts)
            times.append(time.time() - start)
            counter += 1

        except GeneratorExit:
            print("[INFO] Image processing coroutine closed.")
            return
        except Exception as e:
            print("[ERROR] Exception in image processing:")
            traceback.print_exc()
            continue

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
        if event.is_directory:
            return
        elif event.event_type == "created":
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
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDirectory, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(300)
                UDPSocket.sendto(np.array([0.0]).tobytes(), (hostnamePC, 8888))
                self.observer.stop()
                print("Observer Stopped")
                break
        except:
            UDPSocket.sendto(np.array([0.0]).tobytes(), (hostnamePC, 8888))
            self.observer.stop()
            print("Observer Interrupted")
        self.observer.join()

# -------------------------------
# Main loop
# -------------------------------
if __name__ == "__main__":
    watch = OnMyWatch()
    watch.run()
    if len(times):
        times = np.array(times[1:])
        print(f"[RESULTS] Processing at {round(1 / np.mean(times), 2)} FPS")
        print(f"[RESULTS] {len(times)} valid images")
    else:
        print("[RESULTS] No valid images captured")

    print("Display frames with OpenCV...")
    for fid, frame in enumerate(frames):
        print(f" Displaying Frame {fid}")
        cv2.imshow("Blobs", frame)
        cv2.waitKey(10)

    cv2.destroyAllWindows()