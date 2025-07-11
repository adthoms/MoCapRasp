import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the image containing the ArUco marker board
image_path = "marker_board.jpg"

# Parameters for the board
SQUARES_VERTICALLY = 6
SQUARES_HORIZONTALLY = 6
MARKER_LENGTH = 0.0880  # Marker length in meters
BUFFER_LENGTH = 0.0264  # Buffer length in meters

# List of standard ArUco dictionaries to try
#  We will iterate through these dictionaries.
#  Note that DICT_APRILTAG_ are often used for pure ArUco tags,
#   while DICT_4X4, DICT_5X5, etc. are common for ChArUco boards.
DICT_LIST = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_4X4_1000", cv2.aruco.DICT_4X4_1000),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_5X5_1000", cv2.aruco.DICT_5X5_1000),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_6X6_1000", cv2.aruco.DICT_6X6_1000),
    ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
    ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
    ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
    ("DICT_7X7_1000", cv2.aruco.DICT_7X7_1000),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ("DICT_APRILTAG_16h5", cv2.aruco.DICT_APRILTAG_16h5),
    ("DICT_APRILTAG_25h9", cv2.aruco.DICT_APRILTAG_25h9),
    ("DICT_APRILTAG_36h10", cv2.aruco.DICT_APRILTAG_36h10),
    ("DICT_APRILTAG_36h11", cv2.aruco.DICT_APRILTAG_36h11),
]

if __name__ == "__main__":
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found_solution = False
    output_image = None
    successful_dict_name = None

    print("Attempting to detect board using all available dictionaries...")

    # Iterate through all dictionaries
    for dict_name, dict_constant in DICT_LIST:
        print(f"\n Trying Dictionary: {dict_name}")

        # Get the dictionary object
        try:
            aruco_dictionary = cv2.aruco.getPredefinedDictionary(dict_constant)
        except Exception as e:
            print(f"Error loading dictionary {dict_name}: {e}. Skipping.")
            continue

        # Create the detector
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dictionary, detector_params)

        # Detect ArUco markers
        marker_corners, marker_ids, rejected_img_points = detector.detectMarkers(gray)

        # Check if we have detected markers
        if marker_ids is None or len(marker_ids) == 0:
            print(f"No markers detected with {dict_name}.")
            continue
        if len(marker_ids) < 0.50 * SQUARES_VERTICALLY * SQUARES_HORIZONTALLY:
            print(f"Detected {len(marker_ids)} ArUco markers with {dict_name}.")
            continue
        print(f"Detected {len(marker_ids)} ArUco markers with {dict_name}.")
        found_solution = True
        successful_dict_name = dict_name

        # Prepare image and draw markers
        output_image = image.copy()
        cv2.aruco.drawDetectedMarkers(output_image, marker_corners, marker_ids)

        # Show detected markers
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title(f"{dict_name} – {len(marker_ids)} ArUco markers detected")
        plt.axis("off")
        plt.show()

    # Final Output
    if found_solution:
        output_filename = f"detected_board_{successful_dict_name}.jpg"
        cv2.imwrite(output_filename, output_image)
        print(
            f"\nDetection complete and successful using {successful_dict_name}. Result saved as '{output_filename}'"
        )
    else:
        print("\nFailed to detect the board using any of the available dictionaries.")
