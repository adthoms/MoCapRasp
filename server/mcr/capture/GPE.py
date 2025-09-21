import os
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from cv2 import destroyAllWindows, triangulatePoints

from mcr.misc.math import findPlane
from mcr.misc.plot import ArenaViewer
from mcr.misc.cameras import projectionPoints
from mcr.misc.markers import processCentroids, getOrderPerEpiline
from mcr.capture.CaptureProcess import CaptureProcess, CalibrationResult

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


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
        self.saved_data_rows, self.undistorted_frames = [], []
        self.calibration_result = CalibrationResult()

    def collect(self):
        """
        Collects 2D marker data from all cameras via UDP.

        This method waits for the first valid marker message from each camera, undistorts
        the points using intrinsic parameters, and stores them in memory for later processing.
        Optionally saves raw data to disk if `self.save` is True.
        """
        log.info("Starting Ground Plane Estimation (GPE) process...")

        # Internal variables
        capture, counter = np.ones(self.cameras, dtype=bool), np.zeros(
            self.cameras, dtype=np.int8
        )

        for _ in range(self.cameras):  # For each camera
            self.undistorted_frames.append([])  # List for each camera

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
                            self.saved_data_rows.append(
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
                            self.undistorted_frames[idx] = np.hstack(
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
                    np.array(self.saved_data_rows),
                    delimiter=",",
                )
            log.info(
                "Saved raw data for GPE process. Estimation not performed in --collect mode."
            )

    def estimate(self, datapath: str) -> None:
        """
        Estimates the ground plane using previously saved marker data.

        Args:
            datapath (str): Path to the CSV file containing 2D marker data (one row per camera).
        """
        # Load calibration files
        self._load_calibration_files()
        # Order centroids
        self._order_epipolar_points()
        # Calculate all points in 3D
        self.all_points_3d = self._triangulate_3d_points()
        # Align the ground plane to the estimated 3D points
        plane_rotation, plane_distance, plane_y_coeff, camera_height = (
            self._align_to_ground_plane(
                self.all_points_3d, self.calibration_result.projection_matrices
            )
        )
        # Plotting the arena
        self._plot_arena(
            plane_rotation=plane_rotation,
            plane_distance=plane_distance,
            plane_y_coeff=plane_y_coeff,
            camera_height=camera_height,
        )
        # Save aligned data
        self._save_aligned_data(
            plane_rotation=plane_rotation,
            ground_data=np.array([plane_distance, plane_y_coeff, camera_height]),
        )

    def _load_calibration_files(self):
        """
        Loads calibration files from disk into the internal CalibrationResult structure.

        Expects the following files in `mcr/capture/data/`:
            - R.csv: rotation matrices
            - t.csv: translation vectors
            - projMat.csv: projection matrices
            - lamb.csv: scale factors
            - F.csv: fundamental matrices
        """
        log.info(f"Loading calibration results from ./mcr/capture/data/")
        self.calibration_result.rotations = np.genfromtxt(
            "mcr/capture/data/R.csv", delimiter=","
        ).reshape(-1, 3, 3)
        self.calibration_result.translations = np.genfromtxt(
            "mcr/capture/data/t.csv", delimiter=","
        ).reshape(-1, 1, 3)
        self.calibration_result.projection_matrices = np.genfromtxt(
            "mcr/capture/data/projMat.csv", delimiter=","
        ).reshape(-1, 4, 4)
        self.calibration_result.scale = np.genfromtxt(
            "mcr/capture/data/lamb.csv", delimiter=","
        )
        self.calibration_result.fundamental_matrices = np.genfromtxt(
            "mcr/capture/data/F.csv", delimiter=","
        ).reshape(-1, 3, 3)

    def _order_epipolar_points(self) -> None:
        """
        Reorders the undistorted marker coordinates across camera pairs
        based on epipolar geometry to ensure correct correspondence.

        Uses the fundamental matrix between each camera pair to determine
        the optimal ordering of points in the second view relative to the first.
        """
        for j in range(self.cameras - 1):
            # Order centroids per epipolar line
            pts1 = self.undistorted_frames[j][0:6].reshape(-1, 2)
            pts2 = self.undistorted_frames[j + 1][0:6].reshape(-1, 2)
            orderSecondFrame = getOrderPerEpiline(
                pts1, pts2, 3, np.copy(self.calibration_result.fundamental_matrices[j])
            )
            pts2 = np.copy(pts2[orderSecondFrame])
            # Save dataset
            self.undistorted_frames[j + 1][0:6] = pts2.copy().ravel()

    def _triangulate_3d_points(self) -> np.ndarray:
        """
        Triangulates 3D points using the first two undistorted camera frames.

        Projects corresponding 2D marker coordinates into 3D using known
        camera intrinsics and extrinsics. Applies a scale correction from calibration.

        Returns:
            np.ndarray: Triangulated 3D marker positions in the world coordinate frame.
        """
        # Triangulate ordered centroids from the first pair
        pts1 = np.copy(self.undistorted_frames[0][0:6].reshape(-1, 2))
        pts2 = np.copy(self.undistorted_frames[1][0:6].reshape(-1, 2))

        # Prepare projection matrices
        P1 = np.hstack((self.camera_matrix[0], [[0.0], [0.0], [0.0]]))
        P2 = np.matmul(
            self.camera_matrix[1],
            np.hstack(
                (
                    self.calibration_result.rotations[1],
                    self.calibration_result.translations[1].reshape(3, 1),
                )
            ),
        )

        # Project points onto the image plane
        projPt1 = projectionPoints(np.array(pts1))
        projPt2 = projectionPoints(np.array(pts2))

        # Triangulate points
        points4d = triangulatePoints(
            P1.astype(float),
            P2.astype(float),
            projPt1.astype(float),
            projPt2.astype(float),
        )
        # Convert to 3D points
        points3d = (points4d[:3, :] / points4d[3, :]).T
        if points3d[0, 2] < 0:
            points3d = -points3d

        return points3d * self.calibration_result.scale[1] / 100

    def _align_to_ground_plane(self, all_points_3d: np.ndarray) -> tuple:
        """
        Computes the transformation required to align the estimated plane to the canonical ground frame.

        Args:
            all_points_3d (np.ndarray): Triangulated 3D marker coordinates.

        Returns:
            tuple:
                - plane_rotation (np.ndarray): 4x4 rotation matrix aligning the plane to ground.
                - camera_height (float): Height of camera 0 after transformation.
                - plane_distance (float): Plane distance from origin.
                - plane_y_coeff (float): Original y-axis coefficient of the plane.
        """
        # Fit the ground plane to the triangulated points
        plane_coeffs = findPlane(all_points_3d[0], all_points_3d[1], all_points_3d[2])
        if np.any(plane_coeffs[0:3] < 0):
            plane_coeffs = findPlane(
                all_points_3d[0], all_points_3d[2], all_points_3d[1]
            )
        a, b, c, d = plane_coeffs
        plane_distance = d
        plane_y_coeff = b

        # Get the orthonogal vector to the plane
        normal_vec, y_axis = np.array([a, b, c]), np.array([0, 1, 0])
        # Compute the angle between the plane and the y axis
        cos_phi = np.dot(normal_vec, y_axis) / (
            np.linalg.norm(normal_vec) * np.linalg.norm(y_axis)
        )
        sin_phi = np.sqrt(1 - cos_phi**2)
        # Compute the versors
        rotation_axis = np.cross(normal_vec, y_axis)
        rotation_axis /= np.linalg.norm(rotation_axis)
        ux, uy, uz = rotation_axis

        # Get the rotation matrix and new ground plane coefficients
        R = np.array(
            [
                [
                    cos_phi + ux * ux * (1 - cos_phi),
                    ux * uy * (1 - cos_phi) - uz * sin_phi,
                    uy * sin_phi + ux * uz * (1 - cos_phi),
                ],
                [
                    ux * uy * (1 - cos_phi) + uz * sin_phi,
                    cos_phi + uy * uy * (1 - cos_phi),
                    uy * uz * (1 - cos_phi) - ux * sin_phi,
                ],
                [
                    ux * uz * (1 - cos_phi) - uy * sin_phi,
                    uy * uz * (1 - cos_phi) + ux * sin_phi,
                    cos_phi + uz * uz * (1 - cos_phi),
                ],
            ]
        )

        # Preparing ground plane coefficients
        plane_rotation = np.eye(4)
        plane_rotation[:3, :3] = R

        # Compute the height of the ground plane
        origin = np.array([0, 0, 0, 1])
        displacement = np.array([0, plane_distance / plane_y_coeff, 0, 0])
        cam0_world = np.matmul(self.calibration_result.projection_matrices[0], origin)
        cam0_ground = np.matmul(plane_rotation, cam0_world + displacement)
        camera_height = cam0_ground[2]

        return plane_rotation, plane_distance, plane_y_coeff, camera_height

    def _plot_arena(
        self,
        plane_rotation: np.ndarray,
        plane_distance: float,
        plane_y_coeff: float,
        camera_height: float,
    ) -> None:
        """
        Visualizes the aligned marker positions in a 3D plot.

        Args:
            plane_rotation (np.ndarray): 4x4 rotation matrix aligning the plane to ground.
            plane_distance (float): Distance of the plane from the origin.
            plane_y_coeff (float): Original y-axis coefficient of the plane.
            camera_height (float): Height of the camera after transformation.
        """
        self.all_points_3d = np.hstack(
            (self.all_points_3d, np.ones((self.all_points_3d.shape[0], 1)))
        )
        self.all_points_3d += [0, plane_distance / plane_y_coeff, 0, 0]
        self.all_points_3d = np.matmul(plane_rotation, self.all_points_3d.T).T
        self.all_points_3d += [0, 0, -camera_height, 0]
        self.all_points_3d = self.all_points_3d.T

        viewer = ArenaViewer(
            title="Ground Plane Estimation",
            arenaSize=10,
            plotSize=(900, 700),
            reference=True,
            graphical=False,
        )
        # TODO: Add camera data to the viewer
        viewer.plot()
        viewer.show()

    def _save_aligned_data(
        self, plane_rotation: np.ndarray, ground_data: np.ndarray
    ) -> None:
        """
        Saves aligned ground plane transformation and scalar parameters to CSV.

        Args:
            plane_rotation (np.ndarray): 4x4 matrix rotating world to ground-aligned frame.
            ground_data (np.ndarray): Array of [plane_distance, plane_y_coeff, camera_height].
        """
        np.savetxt(
            "mcr/capture/data/P_plane.csv", np.array(plane_rotation), delimiter=","
        )
        np.savetxt(
            "mcr/capture/data/groundData.csv", np.array(ground_data), delimiter=","
        )
