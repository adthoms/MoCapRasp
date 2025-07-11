import os, sys
import cv2
import glob
import numpy as np

try:
    from constants import (
        MARKER_LENGTH,
        SQUARE_LENGTH,
        SQUARES_HORIZONTALLY,
        SQUARES_VERTICALLY,
    )
except ImportError:
    print("Error: constants.py not found. Make sure it is in the same directory.")
    sys.exit(1)

DICT = cv2.aruco.DICT_APRILTAG_36h11

# Load predefined dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Define grid board object
board = cv2.aruco.GridBoard(
    (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict,
)

# Accumulate detected points using Python lists
# This is crucial: all_corners will be a list of arrays (InputArrayOfArrays)
all_corners = np.array([], dtype=np.float32).reshape(0, 2)
# all_ids_flat will be a single, concatenated array of all IDs from all images
all_ids_flat = np.array([], dtype=np.int32)
# num_detected_markers_per_image will store the count for each successfully processed image
num_detected_markers_per_image = np.array([], dtype=np.int32)
image_size = None

# load images
camera_number = 1
# For real calibration, use glob.glob to get all images captured by capture.py
images = glob.glob(os.path.expanduser(f"./pics/cam{camera_number}/*.jpg"))

for fname in images:
    print(f"Processing {fname}")
    img = cv2.imread(fname)
    if img is None:
        print(f"Error loading image: {fname}")
        continue

    # Convert to grayscale for marker detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers in the image
    c, i, rejectedImgPoints = detector.detectMarkers(gray)
    c = np.array(c, dtype=np.float32) if c is not None else None
    i = np.array(i, dtype=np.int32) if i is not None else None

    if i is not None and len(i) > 0:
        # Append the corners detected in THIS image as a single element to all_corners
        all_corners = np.vstack((all_corners, c)) if all_corners.size else c
        # Extend the flattened IDs from THIS image to the overall flat list
        all_ids_flat = (
            np.concatenate((all_ids_flat, i.flatten()))
            if all_ids_flat.size
            else i.flatten()
        )
        # Store the count of markers for THIS image
        num_detected_markers_per_image = np.append(
            num_detected_markers_per_image, len(i)
        )
        print(f"Detected {len(i)} markers in {fname}")

        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        # Draw and show detected markers
        img_display = img.copy()  # Create a copy to draw on
        cv2.aruco.drawDetectedMarkers(img_display, c, i)
        cv2.imshow(
            f"Detected {len(i)} Markers in {os.path.basename(fname)}", img_display
        )
        cv2.waitKey(500)
    else:
        print(
            f"No markers detected in {fname}. This image will be skipped for calibration."
        )

cv2.destroyAllWindows()

# Prepare data for calibration
# Convert lists to NumPy arrays with appropriate dtypes
all_corners = [np.array(c, dtype=np.float32) for c in all_corners]
# all_ids_np will be the single, concatenated array of all IDs
all_ids_np = np.array(all_ids_flat, dtype=np.int32)
# num_detected_markers_np will be the counter array
num_detected_markers_np = np.array(num_detected_markers_per_image, dtype=np.int32)

# Camera calibration
# Calibration requires at least a few views (e.g., 4 or more)
if (
    len(all_corners) < 4
):  # Changed from 0 to 4 as typically required for good calibration
    print(
        "Not enough images with detected markers for calibration (need at least 4). Calibration cannot be performed."
    )
    exit()

# Initialize cameraMatrix and distCoeffs as empty NumPy arrays for output.
# These will be filled by the calibrateCameraAruco function.
camera_matrix_init = np.zeros((3, 3), dtype=np.float64)
dist_coeffs_init = np.zeros((1, 5), dtype=np.float64)  # Common for K1, K2, P1, P2, K3

# Call cv2.aruco.calibrateCameraAruco with all required arguments
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
    corners=all_corners,  # List of arrays, one array per image
    ids=all_ids_np,  # Single concatenated array of all IDs
    counter=num_detected_markers_np,  # Array indicating marker count per image
    board=board,  # The defined GridBoard object
    imageSize=image_size,  # Size of the images
    cameraMatrix=None,  # Initialized output camera matrix
    distCoeffs=None,  # Initialized output distortion coefficients
)

if ret:
    print("\nCamera calibration successful.")
    print(f"RMS Error: {ret}")
    print("Camera matrix : \n")
    print(camera_matrix)
    print("Distortion coefficients : \n")
    print(dist_coeffs)

    # Save results to a NumPy .npz file
    np.savez(
        "calibration_results_aruco.npz",
        mtx=camera_matrix,
        dist=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
    )
    print("Calibration results saved to calibration_results_aruco.npz")
else:
    print("\nCamera calibration failed.")
