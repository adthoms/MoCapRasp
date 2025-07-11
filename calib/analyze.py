import cv2
import numpy as np
import glob

# Chessboard size
checkerboard_size = (9, 9)  # inner corners (adapt if your board is different)

# prepare object points
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(
    -1, 2
)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# load images
images = glob.glob(os.path.expanduser("~/calibration_images/*.jpg"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Optional: draw and show the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow("img", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera matrix : \n")
print(mtx)
print("Distortion coefficients : \n")
print(dist)
