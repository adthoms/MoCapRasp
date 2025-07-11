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
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.minMarkerPerimeterRate = 0.01
parameters.maxErroneousBitsInBorderRate = 0.5
parameters.errorCorrectionRate = 0.9
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Define grid board object
board = cv2.aruco.GridBoard(
    (SQUARES_HORIZONTALLY, SQUARES_VERTICALLY),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict,
)

# load images
camera_numbers = [int(num) for num in sys.argv[1:] if num.isdigit()]

for cam_num in camera_numbers:
    images = glob.glob(f"./pics/cam{cam_num}/calib_*.jpg")
    # Accumulate detected points using Python lists
    
    all_corners = np.array([], dtype=np.float32).reshape(0, 2)  # 2D Corners
    all_ids = np.array([], dtype=np.int32)  # Flattened IDs
    counter = np.array([], dtype=np.int32)   # Count of markers per image
    image_size = None

    for fname in images:
        print(f"Processing {fname}")
        img = cv2.imread(fname)
        if img is None:
            print(f"Error loading image: {fname}")
            continue

        # Convert to grayscale if needed
        gray = img.copy() if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce IR noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Improve contrast
        gray = cv2.equalizeHist(gray)

        # Adaptive thresholding for uneven IR lighting
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Detect markers in the image
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        corners = np.array(corners, dtype=np.float32) if corners is not None else None
        ids = np.array(ids, dtype=np.int32) if ids is not None else None

        if ids is not None and len(ids) > 0:
            # Append the corners detected in THIS image as a single element to all_corners
            all_corners = np.vstack((all_corners, corners)) if all_corners.size else corners
            # Extend the flattened IDs from THIS image to the overall flat list
            all_ids = (
                np.concatenate((all_ids, ids.flatten()))
                if all_ids.size
                else ids.flatten()
            )
            # Store the count of markers for THIS image
            counter = np.append(counter, len(ids))
            print(f"Detected {len(ids)} markers in {fname}")

            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)

            # Draw and show detected markers
            img_display = img.copy()  # Create a copy to draw on
            cv2.aruco.drawDetectedMarkers(img_display, corners, ids)
            # cv2.imshow(
            #     f"Detected {len(ids)} Markers in {os.path.basename(fname)}", img_display
            # )
            # cv2.waitKey(300)

    cv2.destroyAllWindows()
    # Prepare data for calibration
    all_corners = np.array(all_corners, dtype=np.float32)
    all_ids = np.array(all_ids, dtype=np.int32)
    counter = np.array(counter, dtype=np.int32)
    print(all_corners.shape, all_ids.shape, counter)

    # Camera calibration
    # Calibration requires at least a few views (e.g., 4 or more)
    if all_corners.size < 4 or all_ids.size < 4:
        print(
            f"Not enough images with detected markers for calibration (need at least 4). \nCamera calibration for cam{cam_num} failed."
        )
        continue

    # Call cv2.aruco.calibrateCameraAruco with all required arguments
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners=all_corners,  # List of arrays, one array per image
        ids=all_ids,  # Single concatenated array of all IDs
        counter=counter,  # Array indicating marker count per image
        board=board,  # The defined GridBoard object
        imageSize=image_size,  # Size of the images
        cameraMatrix=None,  # Initialized output camera matrix
        distCoeffs=None,  # Initialized output distortion coefficients
    )

    if ret:
        print(f"\nCamera calibration for cam{cam_num} successful.")
        print(f"RMS Error: {ret}")
        print("Camera matrix :")
        print(camera_matrix)
        print("\nDistortion coefficients :")
        print(dist_coeffs)

        # Save results to a NumPy .npz file
        np.savez(
            f"calibration_results_cam{cam_num}.npz",
            mtx=camera_matrix,
            dist=dist_coeffs,
            rvecs=rvecs,
            tvecs=tvecs,
        )
        print(f"\nCalibration results saved to calibration_results_cam{cam_num}.npz")
    else:
        print(f"\nCamera calibration for cam{cam_num} failed.")
