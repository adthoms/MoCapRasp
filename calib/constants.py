import cv2

# Path to the image containing the ArUco marker board
image_path = "marker_board.jpg"

# Parameters for the board
SQUARES_VERTICALLY = 6
SQUARES_HORIZONTALLY = 6
MARKER_LENGTH = 0.0880  # Marker length in meters
BUFFER_LENGTH = 0.0264  # Buffer length in meters

# Calculate the Square Length for the ChArUco board
SQUARE_LENGTH = MARKER_LENGTH + BUFFER_LENGTH

# List of standard ArUco dictionaries to try
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
