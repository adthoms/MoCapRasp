from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import cv2, os, socket, time, argparse
import traceback

# parser for command line
parser = argparse.ArgumentParser(
    description="""Image processing client for the MoCap system at the Erobotica lab of UFCG.
                                                \nPlease use it together with the corresponding server script.""",
    add_help=False,
)
parser.add_argument(
    "-min",
    type=int,
    help="minimal threshold for the blob detector (default: 50)",
    default=50,
)
parser.add_argument(
    "-max",
    type=int,
    help="maximal threshold for the blob detector (default: 151)",
    default=151,
)
parser.add_argument(
    "-s", type=int, help="step between the threshold filters (default: 50)", default=50
)
parser.add_argument(
    "-high",
    type=int,
    help="high threshold filter cut-off intensity (to identify area used by the processor)",
    default=127,
)
parser.add_argument(
    "-rep", type=int, help="minimal repeatability for the blobs", default=3
)
parser.add_argument("-area", type=float, help="minimal area for the blobs", default=2)
parser.add_argument(
    "--help",
    action="help",
    default=argparse.SUPPRESS,
    help="Show this help message and exit.",
)
args = parser.parse_args()

os.system("rm -rf /dev/shm/*.bmp")

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = args.min
params.thresholdStep = args.s
params.maxThreshold = args.max
params.minDistBetweenBlobs = 0
params.filterByColor = True
params.filterByArea = True
params.minArea = args.area
params.filterByConvexity = False
params.minConvexity = 0.90
params.blobColor = 0
params.minRepeatability = args.rep
detector = cv2.SimpleBlobDetector_create(params)

pid = os.getpid()
os.system("sudo renice -n -19 -p " + str(pid))
times = []
frames = []

# Socket parameters
UDPSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
hostnamePC = socket.gethostbyname("nuc.local")


def imageProcessing():
    counter = 0
    bitsShift, constMultiplier, highThresh = 4, 16, args.high
    print("Starting image processing loop...")
    while True:
        try:
            print("Waiting for new image...")
            start = time.time()

            img, ts = yield
            print(
                f"Received image at timestamp {ts} with shape {None if img is None else img.shape}"
            )

            _, thresh = cv2.threshold(img, highThresh, 255, cv2.THRESH_BINARY)
            if img is None or img.size == 0:
                print("[DEBUG] Received an empty image, skipping...")
                continue

            coord = cv2.findNonZero(thresh)
            if coord is None or coord.size == 0:
                print("[DEBUG] No non-zero coordinates found in thresholded image.")
                continue

            coord = coord.reshape(-1, 2).T
            xMin, xMax = int(min(coord[1])), int(max(coord[1]))
            yMin, yMax = int(min(coord[0])), int(max(coord[0]))
            print(f"[DEBUG] Bounding box: x=({xMin}, {xMax}), y=({yMin}, {yMax})")

            x1 = max(0, xMin - 5)
            x2 = min(img.shape[0], xMax + 5)
            y1 = max(0, yMin - 5)
            y2 = min(img.shape[1], yMax + 5)

            cropped = img[x1:x2, y1:y2]
            if cropped.size == 0:
                print(
                    f"[DEBUG] Cropped region is empty: shape {cropped.shape}, skipping."
                )
                continue

            keypoints = detector.detect(cv2.bitwise_not(cropped))
            N = len(keypoints)
            msg = np.zeros(N * 3 + 4)
            for i in range(N):
                msg[(i << 1) + i], msg[(i << 1) + i + 1], msg[(i << 1) + i + 2] = (
                    keypoints[i].pt[0],
                    keypoints[i].pt[1],
                    keypoints[i].size,
                )
            msg[-4], msg[-3], msg[-2], msg[-1] = xMin, yMin, ts, counter
            print("attempting to send...")
            UDPSocket.sendto(msg.tobytes(), (hostnamePC, 8888))
            print("send successful.")
            imgWithKPts = cv2.cvtColor(
                img[xMin - 10 : xMax + 10, yMin - 10 : yMax + 10], cv2.COLOR_GRAY2BGR
            )
            for keyPt in keypoints:
                center = (
                    int(np.round(keyPt.pt[0] * constMultiplier)),
                    int(np.round(keyPt.pt[1] * constMultiplier)),
                )
                radius = int(np.round(keyPt.size / 2 * constMultiplier))
                imgWitframeshKPts = cv2.circle(
                    imgWithKPts,
                    center,
                    radius,
                    (255, 0, 0),
                    thickness=1,
                    lineType=16,
                    shift=bitsShift,
                )
                cv2.circle(imgWithKPts, center, 1, (0, 0, 255), -1, shift=bitsShift)
            frames.append(imgWithKPts)
            times.append(time.time() - start)
            counter += 1
        except GeneratorExit:
            print("[ERROR] Exception in image processing:")
            traceback.print_exc()
            return
        except:
            print("[ERROR] Exception in image processing:")
            traceback.print_exc()
            continue


class OnMyWatch:
    # Set the directoryframes on watch
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
                self.observer.stop()
                UDPSocket.sendto(np.array([0.0]).tobytes(), (hostnamePC, 8888))
                print("Observer Stopped 1")
                break
        except:
            UDPSocket.sendto(np.array([0.0]).tobytes(), (hostnamePC, 8888))
            self.observer.stop()
            print("Observer Stopped 2")
        self.observer.join()


class Handler(FileSystemEventHandler):
    counter = 0
    coRout = imageProcessing()
    coRout.__next__()
    lastImg = ""

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None
        elif event.event_type == "created":
            # print("Watchdog received created event - % s." % event.src_path)
            if Handler.counter:
                name = Handler.lastImg
                img = cv2.imread("/dev/shm/" + name + ".bmp", cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    Handler.coRout.send((img, int(name)))
                os.remove("/dev/shm/" + name + ".bmp")
            Handler.lastImg = event.src_path[-14:-4]
            Handler.counter += 1


if __name__ == "__main__":
    watch = OnMyWatch()
    watch.run()
    if len(times):
        times = np.array(times[1:])
        print("[RESULTS] processing with " + str(round(1 / np.mean(times), 2)) + "FPS")
        print("[RESULTS] " + str(len(times)) + " valid images")
    else:
        print("[RESULTS] no valid images captured")
    print("Display frames with OpenCV...")
    for fid, frame in enumerate(frames):
        print(" Displaying Frame ", fid)
        cv2.imshow("Slow Motion", frame)
        cv2.waitKey(10)  # request maximum refresh rate

    cv2.destroyAllWindows()
