import os
import sys
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

from constants import (
    MARKER_LENGTH,
    BUFFER_LENGTH,
    SQUARES_HORIZONTALLY,
    SQUARES_VERTICALLY,
)

DICT = cv2.aruco.DICT_APRILTAG_36h11
MIN_MARKERS = 12    # Minimum number of markers to consider a valid calibration

def get_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.aprilTagQuadDecimate = 1.0
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 101
    params.adaptiveThreshWinSizeStep = 3
    params.maxErroneousBitsInBorderRate = 0.6
    params.polygonalApproxAccuracyRate = 0.04
    params.errorCorrectionRate = 0.8
    return cv2.aruco.ArucoDetector(aruco_dict, params), aruco_dict

def get_board(aruco_dict):
    return cv2.aruco.GridBoard(
        size=(SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
        markerLength=MARKER_LENGTH,
        markerSeparation=BUFFER_LENGTH,
        dictionary=aruco_dict,
    )

def load_images(cam_num):
    images = glob.glob(f"./pics/cam{cam_num}/calib_*.jpg")
    images.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    return images

def process_image(img_path, detector, display=False, verbose=False):
    img = cv2.imread(img_path)
    if img is None:
        if verbose:
            print(f"Error loading image: {img_path}")
        return None, None, None

    gray = img.copy() if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) >= MIN_MARKERS:
        if verbose:
            print(f"{img_path} handled: num of detected markers = {len(ids)}")
        if display:
            vis = gray.copy()
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            cv2.imshow(f"Markers in {os.path.basename(img_path)}", vis)
            cv2.waitKey(300)
        return corners, ids, gray.shape[::-1]
    else:
        if verbose:
            print(f"{img_path} skipped: num of detected markers = {len(ids) if ids is not None else 0}")
        return None, None, None

def calibrate_camera(corners_list, ids_list, counter, image_size, board):
    if len(corners_list) < 4:
        return None

    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners=corners_list,
        ids=ids_list,
        counter=counter,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6,
    )
    return ret, mtx, dist, rvecs, tvecs

def plot_corner_heatmap(corners_list, image_size, output_path="corner_heatmap.png"):
    heat = np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    for marker_corners in corners_list:
        for marker in marker_corners:
            for (x, y) in marker:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < heat.shape[1] and 0 <= yi < heat.shape[0]:
                    heat[yi, xi] += 1

    plt.figure(figsize=(8, 6))
    plt.title("Detected Marker Corner Heatmap")
    plt.imshow(heat, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Corner Hits")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_results(cam_num, mtx, dist, rvecs, tvecs):
    filename = f"cam{cam_num}_calib.npz"
    np.savez(filename, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Calibration results saved to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cameras", nargs="+", type=int, help="Camera numbers")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--heatmap", action="store_true")
    args = parser.parse_args()

    detector, aruco_dict = get_detector()
    board = get_board(aruco_dict)
    print(f"Using ArUco dictionary: {DICT} with {board.getGridSize()} grid of markers.")

    for cam_num in args.cameras:
        print(f"\nProcessing camera {cam_num}...")
        images = load_images(cam_num)
        all_corners, all_ids, counter = [], [], []
        image_size = None

        for fname in images:
            corners, ids, size = process_image(fname, detector, display=args.display, verbose=args.verbose)
            if corners is not None:
                all_corners.extend(corners)
                all_ids.extend(ids.flatten())
                counter.append(len(ids))
                if image_size is None:
                    image_size = size

        cv2.destroyAllWindows()

        if args.heatmap and image_size and all_corners:
            plot_corner_heatmap(all_corners, image_size, f"cam{cam_num}_heatmap.png")
        if len(all_corners) < 4:
            print(f"Not enough data to calibrate camera {cam_num}. Skipping.")
            continue

        result = calibrate_camera(all_corners, np.array(all_ids), np.array(counter), image_size, board)

        if result is None:
            print(f"Calibration failed for camera {cam_num}.")
            continue

        ret, mtx, dist, rvecs, tvecs = result
        print(f"RMS Error for camera {cam_num}: {ret}")
        if ret > 1.0:
            print("RMS error too high (>1.0), check marker coverage and focus.")
            continue

        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)

        if args.save:
            save_results(cam_num, mtx, dist, rvecs, tvecs)

if __name__ == "__main__":
    main()
