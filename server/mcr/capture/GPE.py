import os
import logging
import warnings
import numpy as np
from datetime import datetime
from cv2 import destroyAllWindows, triangulatePoints

from mcr.misc.math import findPlane
from mcr.misc.plot import ArenaViewer, Frame
from mcr.misc.cameras import projectionPoints
from mcr.misc.markers import processCentroids, getOrderPerEpiline
from mcr.capture.CaptureProcess import CaptureProcess

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# class CaptureProcess(object):
#     def __init__(
#         self, cameraids, markers, trigger, record, fps, verbose, save, *args, **kwargs
#     ):

# gpeServer = GPE(cameraids, markers, trigger, record, fps, verbose, save)


class GPE(CaptureProcess):
    """
    Ground Plane Estimation (GPE) module.

    This class inherits from CaptureProcess and is responsible for:
    - Capturing 2D marker data from multiple cameras via UDP.
    - Undistorting and storing the first valid observation from each camera.
    - Reordering markers based on epipolar geometry using the fundamental matrix.
    - Triangulating 3D points from the 2D correspondences.
    - Fitting a plane to the triangulated points and aligning it to a canonical ground frame.
    - Saving aligned 3D data and camera height relative to the estimated ground plane.
    """

    def __init__(
        self, cameraids, markers, trigger, record, fps, verbose, save, *args, **kwargs
    ):
        """
        Initializes the GPE instance.

        Args:
            cameraids (list): List of camera IDs.
            markers (list): List of marker IDs.
            trigger (bool): Whether to trigger the capture.
            record (bool): Whether to record the capture.
            fps (int): Frames per second for the capture.
            verbose (bool): Whether to enable verbose logging.
            save (bool): Whether to save the captured data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            cameraids, markers, trigger, record, fps, verbose, save, *args, **kwargs
        )

    # Collect points from clients, order and trigger interpolation
    def collect(self):
        log.info("Starting Ground Plane Estimation (GPE) process...")

        # Internal variables
        capture, counter = np.ones(self.cameras, dtype=bool), np.zeros(
            self.cameras, dtype=np.int8
        )
        saved_data_rows, time_intervals = [], []

        for _ in range(self.cameras):  # For each camera
            time_intervals.append([])  # List for each camera

        # Capture loop
        try:
            while np.any(capture):
                # Receive message
                bytesPair = self.server_socket.recvfrom(self.buffer_size)
                message = np.frombuffer(bytesPair[0], dtype=np.float64)
                address, sizeMsg = bytesPair[1], len(message)
                idx = self.ipList.index(address[0])
                if not (sizeMsg - 1):
                    capture[idx] = 0

                if capture[idx]:  # Check if message is valid
                    msg = message[0 : sizeMsg - 4].reshape(-1, 3)
                    coord = msg[:, 0:2]

                    # Store message parameters
                    a, b, timeNow, imgNumber = (
                        message[-4],
                        message[-3],
                        message[-2],
                        int(message[-1]),
                    )
                    if not len(coord):
                        continue

                    # Undistort points
                    undCoord = processCentroids(
                        coord, a, b, self.camera_matrix[idx], self.distortion_coeff[idx]
                    )
                    if undCoord.shape[0] == 3:
                        if self.save:
                            saved_data_rows.append(
                                np.concatenate(
                                    (
                                        undCoord.reshape(
                                            undCoord.shape[0] * undCoord.shape[1]
                                        ),
                                        [timeNow, imgNumber, idx],
                                    )
                                )
                            )
                        if not counter[idx]:
                            time_intervals[idx] = np.hstack(
                                (undCoord.reshape(6), timeNow)
                            )
                        counter[idx] += 1

                    # Do I have enough points?
                    if np.all(counter > 0):
                        break
        finally:
            # Close everything
            self.server_socket.close()
            destroyAllWindows()

            # Save Ground Plane Estimation (GPE) Data
            if self.save:
                now = datetime.now()
                ymd, HMS = now.strftime("%y-%m-%d"), now.strftime("%H-%M-%S")
                path = "debug/dataSaves/" + ymd + "/"

                # Check whether directory already exists
                if not os.path.exists(path):
                    os.mkdir(path)
                    print(f"Folder {path} created!")

                np.savetxt(
                    path + "GPE-" + HMS + ".csv",
                    np.array(saved_data_rows),
                    delimiter=",",
                )

            # Import R,t and lambda
            rotation = np.genfromtxt("mcr/capture/data/R.csv", delimiter=",").reshape(
                -1, 3, 3
            )
            translation = np.genfromtxt(
                "mcr/capture/data/t.csv", delimiter=","
            ).reshape(-1, 1, 3)
            projMat = np.genfromtxt(
                "mcr/capture/data/projMat.csv", delimiter=","
            ).reshape(-1, 4, 4)
            scale, FMatrix = np.genfromtxt(
                "mcr/capture/data/lamb.csv", delimiter=","
            ), np.genfromtxt("mcr/capture/data/F.csv", delimiter=",").reshape(-1, 3, 3)

            # Order centroids
            for j in range(self.cameras - 1):
                # Collect extrinsics from calibration between camera 0 and 1
                F = FMatrix[j]
                R, t, lamb = (
                    rotation[j + 1],
                    translation[j + 1].reshape(-1, 3),
                    scale[j + 1],
                )

                # Order centroids per epipolar line
                pts1, pts2 = time_intervals[j][0:6].reshape(-1, 2), time_intervals[
                    j + 1
                ][0:6].reshape(-1, 2)
                orderSecondFrame = getOrderPerEpiline(pts1, pts2, 3, np.copy(F))
                pts2 = np.copy(pts2[orderSecondFrame])

                # Save dataset
                time_intervals[j + 1][0:6] = pts2.copy().ravel()

            # Triangulate ordered centroids from the first pair
            pts1, pts2 = np.copy(time_intervals[0][0:6].reshape(-1, 2)), np.copy(
                time_intervals[1][0:6].reshape(-1, 2)
            )
            R, t, lamb = rotation[1], translation[1].reshape(-1, 3), scale[1]
            P1, P2 = np.hstack(
                (self.camera_matrix[0], [[0.0], [0.0], [0.0]])
            ), np.matmul(self.camera_matrix[1], np.hstack((R, t.T)))
            projPt1, projPt2 = projectionPoints(np.array(pts1)), projectionPoints(
                np.array(pts2)
            )
            points4d = triangulatePoints(
                P1.astype(float),
                P2.astype(float),
                projPt1.astype(float),
                projPt2.astype(float),
            )
            points3d = (points4d[:3, :] / points4d[3, :]).T
            if points3d[0, 2] < 0:
                points3d = -points3d
            allPoints3d = points3d * lamb / 100

            # Get ground plane coefficients
            plane = findPlane(allPoints3d[0], allPoints3d[1], allPoints3d[2])
            if np.any(plane[0:3] < 0):
                plane = findPlane(allPoints3d[0], allPoints3d[2], allPoints3d[1])
            [a, b, c, d] = plane

            # Get the orthonogal vector to the plane
            v, k = np.array([a, b, c]), np.array([0, 1, 0])

            # Compute the angle between the plane and the y axis
            cosPhi = np.dot(v, k) / (np.linalg.norm(v) * np.linalg.norm(k))

            # Compute the versors
            [u1, u2, u3] = np.cross(v, k) / np.linalg.norm(np.cross(v, k))

            # Get the rotation matrix and new ground plane coefficients
            sinPhi = np.sqrt(1 - pow(cosPhi, 2))
            R_plane = np.array(
                [
                    [
                        cosPhi + u1 * u1 * (1 - cosPhi),
                        u1 * u2 * (1 - cosPhi) - u3 * sinPhi,
                        u2 * sinPhi + u1 * u3 * (1 - cosPhi),
                    ],
                    [
                        u1 * u2 * (1 - cosPhi) + u3 * sinPhi,
                        cosPhi + u2 * u2 * (1 - cosPhi),
                        u2 * u3 * (1 - cosPhi) - u1 * sinPhi,
                    ],
                    [
                        u1 * u3 * (1 - cosPhi) - u2 * sinPhi,
                        u2 * u3 * (1 - cosPhi) + u1 * sinPhi,
                        cosPhi + u3 * u3 * (1 - cosPhi),
                    ],
                ]
            )
            [A, B, C] = np.matmul(R_plane, np.array([a, b, c]).T)

            # Preparing ground data
            P_plane = np.vstack(
                (
                    np.hstack((R_plane, np.array([0, 0, 0]).reshape(3, -1))),
                    np.hstack((np.zeros((3)), 1)),
                )
            )
            newPlane = np.array([A, B, C])

            groundData = {
                "planeDisplacement": d / b,
                "planeRotation": P_plane,
                "planeCoefficients": newPlane,
            }

            # Preparing camera data
            o = np.matmul(projMat[0], [[0.0], [0.0], [0.0], [1]]).ravel()
            o += [0, d / b, 0, 0]  # Displace in relation to ground
            o = np.matmul(P_plane, o).ravel()  # Rotate in relation to ground
            h = o[2]  # Let 'h' be the height of the 0th camera

            cameraData = {"cameraHeight": h, "projectionMatrices": projMat}

            # Preparing captured data
            allPoints3d = np.hstack((allPoints3d, np.ones((allPoints3d.shape[0], 1))))
            allPoints3d += [0, d / b, 0, 0]
            allPoints3d = np.matmul(P_plane, allPoints3d.T).T
            allPoints3d += [0, 0, -h, 0]
            allPoints3d = allPoints3d.T

            # plotArena(
            #     title="Ground Plane Estimation",
            #     allPoints3d=allPoints3d,
            #     cameraData=cameraData,
            #     groundData=groundData,
            # )

            # Save data
            np.savetxt("mcr/capture/data/P_plane.csv", np.array(P_plane), delimiter=",")
            np.savetxt(
                "mcr/capture/data/groundData.csv", np.array([d, b, h]), delimiter=","
            )
