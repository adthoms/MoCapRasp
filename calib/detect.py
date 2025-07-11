import cv2
import matplotlib.pyplot as plt

from constants import image_path, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, DICT_LIST


def detect_aruco_board(image_path):
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


if __name__ == "__main__":
    detect_aruco_board(image_path)
