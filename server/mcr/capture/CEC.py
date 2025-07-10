import os, math
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
from itertools import combinations, permutations
from cv2 import destroyAllWindows, triangulatePoints

from mcr.capture.CaptureProcess import CaptureProcess, CameraState
from mcr.misc.math import isCollinear
from mcr.misc.cameras import (
    estimateFundMatrix_8norm,
    decomposeEssentialMat,
    projectionPoints,
)
from mcr.misc.markers import orderCenterCoord, occlusion, processCentroids
from mcr.misc.plot import ArenaViewer, Frame


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """
    Holds the full output of the multi-camera calibration process.
    Includes rotation, translation, scaling factors, triangulated points,
    and projection matrices.
    """

    rotations: list = field(
        default_factory=lambda: [np.identity(3)]
    )  # List of relative rotation matrices between camera pairs
    translations: list = field(
        default_factory=lambda: [[[0.0, 0.0, 0.0]]]
    )  # List of relative translations
    scales: list = field(
        default_factory=lambda: [[1]]
    )  # List of scale factors for 3D point normalization
    fundamental_matrices: list = field(
        default_factory=list
    )  # Fundamental matrices (2D epipolar geometry)
    triangulated_points: list = field(
        default_factory=list
    )  # 3D points after triangulation and scale application
    projection_matrices: list = field(
        default_factory=list
    )  # Reserved for projection matrices (optional use)
    all_points_3d: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 0))
    )  # Homogeneous 3D coordinates (shape: 4×N) of all triangulated points,


class CEC(CaptureProcess):
    """
    Camera Extrinsics Calibration (CEC) process.

    Inherits from CaptureProcess and implements a multi-camera calibration
    routine including time-synchronized capture, marker ordering, 2D interpolation,
    essential matrix decomposition, triangulation, and export of camera extrinsics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dbscan_eps = kwargs.get("dbscan_eps", 0.01)
        self.dbscan_min_samples = kwargs.get("dbscan_min_samples", 10)
        self.use_clustering = kwargs.get("use_clustering", False)
        self.camera_states = [CameraState() for _ in range(self.cameras)]
        self.calibration_result = CalibrationResult()

        # Real world distances between markers (in cm)
        self.L_real_AB = 15.00  # Distance between markers A and B
        self.L_real_BC = 20.00  # Distance between markers B and C
        self.L_real_CA = 25.00  # Distance between markers C and A
        self.tolerance = 0.10

        self.expected_ratios = {
            ("AB", "BC"): self.L_real_AB / self.L_real_BC,
            ("BC", "CA"): self.L_real_BC / self.L_real_CA,
            ("CA", "AB"): self.L_real_CA / self.L_real_AB,
            ("BC", "AB"): self.L_real_BC / self.L_real_AB,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable attributes
        if "server_socket" in state:
            state["server_socket"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # You must call `connect()` later to restore the socket

    def collect(self) -> None:
        """
        Main loop to receive data packets from all cameras, process and undistort the
        detected marker blobs, track certainty, and accumulate all valid 2D points.
        When capture is complete, saves the raw data to CSV.
        """
        log.info("Starting capture for Camera Extrinsics Calibration")
        saved_data_rows = []

        try:
            while any(cam.capture_active for cam in self.camera_states):
                bytes_pair = self.server_socket.recvfrom(self.bufferSize)
                message = np.frombuffer(bytes_pair[0], dtype=np.float64)
                address, size_msg = bytes_pair[1], len(message)
                idx = self.ipList.index(address[0])
                self._receive_packet_and_process(
                    idx, self.camera_states[idx], message, size_msg, saved_data_rows
                )
        finally:
            self._finalize_capture_session(saved_data_rows)
            log.info("Saved raw 2D capture data. Calibration not performed in --collect mode.")
            # self._compute_camera_extrinsics()

            # if self.save:
            #     self._save_calibration_results()
            #     log.info("Calibration results saved to CSV files in mcr/capture/data/")

    def calibrate(self, datapath: str) -> None:
        """
        Loads saved 2D marker data from CSV and computes camera extrinsics.

        Args:
            datapath (str): Path to the saved CSV file
        """
        log.info(f"Loading 2D marker data from {datapath}")
        if datapath:
            if not os.path.exists(datapath):
                log.error(f"File not found: {datapath}")
                return
            try:
                data = pd.read_csv(datapath, header=None).values
                for row in data:
                    cam_idx = int(row[-1])
                    if cam_idx >= self.cameras:
                        log.warning(f"Skipping row with invalid camera index {cam_idx}")
                        continue
                    existing = self.camera_states[cam_idx].undistorted_frames
                    if existing is None or np.array(existing).size == 0 or existing.ndim != 2:
                        self.camera_states[cam_idx].undistorted_frames = np.array([row])
                    else:
                        self.camera_states[cam_idx].undistorted_frames = np.vstack([existing, row])
            except Exception as e:
                log.error(f"Failed to load calibration data from file: {e}")
                return

        log.info("Data successfully loaded. Starting calibration process...")
        self._finalize_capture_session(saved_data_rows=None)
        self._compute_camera_extrinsics()
        if self.save:
            self._save_calibration_results()
            log.info("Calibration results saved to CSV files in mcr/capture/data/")

    def _receive_packet_and_process(
        self, idx, cam_state, message, size_msg, saved_data_rows
    ) -> None:
        """
        Processes an incoming packet from a given camera, performing blob area filtering,
        undistortion, occlusion checks, marker ordering, and frame timing consistency.

        Parameters:
            idx (int): Index of the camera
            cam_state (CameraState): Internal state for this camera
            message (np.ndarray): The raw float64 data packet
            size_msg (int): Length of the received message
            saved_data_rows (list): Shared list for storing valid rows for CSV saving
        """
        # Check if the camera is still active
        if not (size_msg - 1):
            cam_state.capture_active = False
            return

        # If less than 3 blobs are found, skip the frame
        if size_msg < 13:
            log.error(f"[CAM{idx}] Only {(size_msg - 4) // 3} markers were found")
            cam_state.missed_frames += 1
            return

        # If more than 3 blobs are found, skip the frame
        if size_msg > 13:
            log.error(f"[CAM{idx}] {(size_msg - 4) // 3} > 3 blobs were found")
            cam_state.missed_frames += 1
            return

        # Reshape the message to extract coordinates and sizes
        msg = message[0 : size_msg - 4].reshape(-1, 3)
        coord, size = msg[:, 0:2], msg[:, 2].reshape(-1)

        # If more than 3 blobs are found, get the three biggest
        # if size_msg > 13:
        #     order = np.argsort(size)[::-1]
        #     coord = coord[order[:3]]

        from scipy.spatial.distance import pdist, squareform

        def remove_close_points(points, min_dist=10.0):
            """
            Filters out points that are closer than min_dist to each other.
            Args:
                points (np.ndarray): Array of points to filter.
                min_dist (float): Minimum distance between points.
            Returns:
                np.ndarray: Filtered array of points.
            """
            if len(points) < 2:
                return points
            dist_matrix = squareform(pdist(points))
            np.fill_diagonal(dist_matrix, np.inf)
            keep = np.ones(len(points), dtype=bool)
            for i in range(len(points)):
                if not keep[i]:
                    continue
                close = np.where(dist_matrix[i] < min_dist)[0]
                keep[close] = False
                keep[i] = True
            return points[keep]

        # Filter out blobs that are too close to each other
        if self.verbose:
            log.debug("")
            log.debug(f"[CAM{idx}] raw coordinates: {coord}")
        coord = remove_close_points(coord, min_dist=15.0)
        log.debug(f"[CAM{idx}] Filtered coord count: {coord.shape[0]}")
        # Now pick up to 3 well-separated blobs
        coord = coord[:3]

        a, b, timestamp, img_number = (
            message[-4],
            message[-3],
            message[-2],
            int(message[-1]),
        )

        und_coord = processCentroids(
            coord, a, b, self.cameraMat[idx], self.distCoef[idx]
        )
        if und_coord.shape != (3, 2):
            log.warning(
                f"[CAM{idx}] Skipping frame due to invalid blob count: {und_coord.shape}"
            )
            cam_state.missed_frames += 1
            cam_state.invalid_frames += 1
            return

        if self.save:
            saved_data_rows.append(
                np.concatenate((und_coord.reshape(6), [timestamp, img_number, idx]))
            )

        # Check timestamp consistency
        if cam_state.frame_counter:
            if abs(timestamp - cam_state.last_timestamp) > 1e9:
                if self.verbose:
                    log.warning(f"[CAM{idx}] Time mismatch")
                cam_state.missed_frames += 1
                cam_state.invalid_frames += 1
                return

        # Check image sequence
        if img_number > cam_state.last_image_id + 1:
            cam_state.invalid_frames = img_number - cam_state.last_image_id

        # Check collinearity, occlusion, and coordinate validity
        if self.verbose:
            log.debug(f"[CAM{idx}] undistorted coordinates: {und_coord}")
            log.debug(f"[CAM{idx}] np.any(undCoord<0): {np.any(und_coord < 0)}")
            log.debug(
                f"[CAM{idx}] isCollinear: {isCollinear(*und_coord)}, occlusion: {occlusion(und_coord, 5)}, invalid: {cam_state.invalid_frames}"
            )

        if (
            not isCollinear(*und_coord)
            and not occlusion(und_coord, 5)
            and not np.any(und_coord < 0)
        ):
            if cam_state.invalid_frames >= 10 or not cam_state.frame_counter:
                if cam_state.has_certainty:
                    beg = cam_state.intervals[-1]
                    end = cam_state.frame_counter - 1
                    cam_state.time_intervals.append(
                        [
                            cam_state.undistorted_frames[beg][6],
                            cam_state.undistorted_frames[end][6],
                        ]
                    )
                    if self.verbose:
                        log.warning(
                            f"[CAM{idx}] Valid from {cam_state.undistorted_frames[beg][6] / 1e6:.2f}s to {cam_state.undistorted_frames[end][6] / 1e6:.2f}s"
                        )
                prev, cam_state.has_certainty = [], False
                cam_state.intervals.append(cam_state.frame_counter)
            else:
                if cam_state.frame_counter == 0:
                    prev = np.array(cam_state.undistorted_frames[0:6]).reshape(-1, 2)
                else:
                    prev = np.array(cam_state.undistorted_frames[-1][0:6]).reshape(
                        -1, 2
                    )

            target_ratios = {
                "ab_ca": self.L_real_AB / self.L_real_CA,
                "bc_ab": self.L_real_BC / self.L_real_AB,
                "ca_bc": self.L_real_CA / self.L_real_BC,
            }
            und_coord, _ = orderCenterCoord(
                und_coord, prev, log=log, target_ratios=target_ratios
            )
            und_coord = np.array(und_coord)

            A, B, C = und_coord
            log.debug(
                f"[CAM{idx}] Marker distances AB={np.linalg.norm(A-B):.2f}, BC={np.linalg.norm(B-C):.2f}, CA={np.linalg.norm(A-C):.2f}"
            )
            log.debug(
                f"[CAM{idx}] Expected Ratios: AB/BC={self.expected_ratios[('AB', 'BC')]:.3f}, BC/CA={self.expected_ratios[('BC', 'CA')]:.3f}, CA/AB={self.expected_ratios[('CA', 'AB')]:.3f}"
            )
            log.debug(
                f"[CAM{idx}] Obtained Ratios: AB/BC={np.linalg.norm(A-B)/np.linalg.norm(B-C):.3f}, BC/CA={np.linalg.norm(B-C)/np.linalg.norm(A-C):.3f}, CA/AB={np.linalg.norm(A-C)/np.linalg.norm(A-B):.3f}"
            )
            log.debug("")
        else:
            if self.verbose:
                log.warning(f"[CAM{idx}] Collinear or occluded coordinates")
            cam_state.missed_frames += 1
            cam_state.invalid_frames += 1
            return

        cam_state.last_timestamp = timestamp
        cam_state.last_image_id = img_number
        cam_state.invalid_frames = 0
        current_row = np.hstack((und_coord.reshape(6), timestamp))

        if cam_state.frame_counter == 0:
            cam_state.undistorted_frames = np.expand_dims(current_row, axis=0)
        else:
            cam_state.undistorted_frames = np.vstack(
                (cam_state.undistorted_frames, current_row)
            )
        cam_state.frame_counter += 1

        # Check marker ordering for certainty
        if not cam_state.has_certainty:
            for [A, B, C] in und_coord.reshape([-1, 3, 2]):
                ab_norm = np.linalg.norm(A - B)
                bc_norm = np.linalg.norm(C - B)

                actual_ratio_ab_bc = ab_norm / bc_norm if bc_norm > 1e-3 else np.inf
                actual_ratio_bc_ab = bc_norm / ab_norm if ab_norm > 1e-3 else np.inf

                log.debug(
                    f"[CAM{idx}] AB={ab_norm:.2f}, BC={bc_norm:.2f}, "
                    f"AB/BC={actual_ratio_ab_bc:.3f} (target: {self.expected_ratios[('AB', 'BC')]:.3f}), "
                    f"BC/AB={actual_ratio_bc_ab:.3f} (target: {self.expected_ratios[('BC', 'AB')]:.3f})"
                )

                # Check if the ratios are within the expected tolerance
                if (
                    abs(actual_ratio_ab_bc - self.expected_ratios[("AB", "BC")])
                    < self.tolerance
                    and ab_norm > 20
                ):
                    cam_state.swap_counter += 1
                    log.debug(
                        f"[CAM{idx}] Swap condition met — swap_counter = {cam_state.swap_counter}"
                    )
                    if cam_state.swap_counter > 2:
                        cam_state.swap_counter = 0
                        cam_state.has_certainty = True
                        start = cam_state.intervals[-1]
                        end = cam_state.frame_counter
                        log.info(
                            f"[CAM{idx}] A-C swap performed between frames {start} and {end}"
                        )
                        (
                            cam_state.undistorted_frames[start:end, 0:2],
                            cam_state.undistorted_frames[start:end, 4:6],
                        ) = np.copy(
                            cam_state.undistorted_frames[start:end, 4:6]
                        ), np.copy(
                            cam_state.undistorted_frames[start:end, 0:2]
                        )

                if (
                    abs(actual_ratio_bc_ab - self.expected_ratios[("BC", "AB")])
                    < self.tolerance
                    and bc_norm > 20
                ):
                    cam_state.has_certainty = True
                    log.info(
                        f"[CAM{idx}] Certainty established without swap (BC/AB matched expected ratio)"
                    )

    def _finalize_capture_session(self, saved_data_rows) -> None:
        """
        After all cameras finish streaming, this method saves the undistorted 2D marker data and
        logs camera summaries.
        """
        try:
            self.server_socket.close()
        except Exception as e:
            log.error(f"Failed to close server socket: {e}")
        destroyAllWindows()

        if self.save and saved_data_rows is not None:
            now = datetime.now()
            ymd, HMS = now.strftime("%y-%m-%d"), now.strftime("%H-%M-%S")
            path = f"debug/dataSaves/{ymd}/"
            if not os.path.exists(path):
                os.makedirs(path)
                log.info(f"Folder {path} created!")
            np.savetxt(
                path + f"CEC-{HMS}.csv", np.array(saved_data_rows), delimiter=","
            )

        # Get last intervals
        log.info("╔══════════════ Camera Synchronization ══════════════╗")
        for idx, state in enumerate(self.camera_states):
            if len(state.undistorted_frames) == 0:
                continue
            if state.has_certainty:
                beg, end = state.intervals[-1], state.frame_counter - 1
                state.time_intervals.append(
                    [state.undistorted_frames[beg, 6], state.undistorted_frames[end, 6]]
                )
                if self.verbose:
                    log.info(
                        f"│ [CAM{idx}] Valid time range : {state.undistorted_frames[beg, 6] / 1e6:6.2f}s — {state.undistorted_frames[end, 6] / 1e6:6.2f}s"
                    )
        log.info("╚════════════════════════════════════════════════════╝")

        if self.verbose:
            log.info("╔═══════════════ Server Results Summary ═══════════════╗")
            for i, state in enumerate(self.camera_states):
                log.info(f"║ CAM{i:<1} Address        : {self.ipList[i]}")
                log.info(f"║       Valid Images      : {len(state.undistorted_frames)}")
                log.info(f"║       Invalid Images    : {state.invalid_frames}")
                log.info(f"║       Missed Images     : {state.missed_frames}")
                log.info(f"║       Intervals         : {state.time_intervals}")
            log.info("╚══════════════════════════════════════════════════════╝")

    def _compute_camera_extrinsics(self):
        """
        Performs pairwise camera extrinsics calibration for all consecutive camera pairs,
        estimates F and E, decomposes into R and t, triangulates, computes scale,
        and stores projection matrices and triangulated points.
        """
        log.debug("")
        log.debug(f"Beginning calibration across {self.cameras - 1} camera pairs")
        for cam in range(self.cameras - 1):
            state1, state2 = self.camera_states[cam], self.camera_states[cam + 1]
            # Compute valid time intersection for interpolation
            intersections = self._get_valid_intersections(state1, state2)

            if self.verbose:
                log.info("")
                log.info(f"Intersection of CAM{cam} and CAM{cam+1}: {intersections}")

            if not intersections:
                continue

            # Interpolate centroids for the overlapping time intervals
            centroids1, centroids2 = self._interpolate_centroids(
                cam, state1, state2, intersections
            )
            if centroids1 is np.nan or centroids2 is np.nan:
                log.error(
                    f"No valid overlap between cameras {cam} and {cam + 1}, skipping calibration"
                )
                continue

            log.info(f"Performing permutation search for camera pair {cam}-{cam+1}")

            # Find best permutation of centroids to match expected ratios
            centroids1 = self._find_best_permutation(cam, centroids1)
            centroids2 = self._find_best_permutation(cam + 1, centroids2)

            # Get fundamental and essential matrices
            log.info(
                f"Computing fundamental and essential matrix between cameras {cam}-{cam+1}"
            )
            log.debug(
                f"[CAM{cam}-{cam+1}] Centroids1 shape: {centroids1.shape}, Centroids2 shape: {centroids2.shape}"
            )
            log.debug(
                f"       Expected Ratios: AB/BC={self.expected_ratios[('AB', 'BC')]:.3f}, BC/CA={self.expected_ratios[('BC', 'CA')]:.3f}, CA/AB={self.expected_ratios[('CA', 'AB')]:.3f}"
            )
            log.debug(
                f"[CAM{cam}] Measured Ratios: AB/BC={np.linalg.norm(centroids1[0]-centroids1[1])/np.linalg.norm(centroids1[1]-centroids1[2]):.3f}, BC/CA={np.linalg.norm(centroids1[1]-centroids1[2])/np.linalg.norm(centroids1[0]-centroids1[2]):.3f}, CA/AB={np.linalg.norm(centroids1[0]-centroids1[2])/np.linalg.norm(centroids1[0]-centroids1[1]):.3f}"
            )
            log.debug(
                f"[CAM{cam+1}] Measured Ratios: AB/BC={np.linalg.norm(centroids2[0]-centroids2[1])/np.linalg.norm(centroids2[1]-centroids2[2]):.3f}, BC/CA={np.linalg.norm(centroids2[1]-centroids2[2])/np.linalg.norm(centroids2[0]-centroids2[2]):.3f}, CA/AB={np.linalg.norm(centroids2[0]-centroids2[2])/np.linalg.norm(centroids2[0]-centroids2[1]):.3f}"
            )
            F, R, t = self._compute_fundamental_essential_and_pose(
                cam, cam + 1, centroids1, centroids2
            )
            if np.isnan(F).any() or np.isnan(R).any() or np.isnan(t).any():
                continue  # Skip this pair if any matrix is invalid
            if self.verbose:
                log.info("Rotation    Matrix\n%s", R.round(4))
                log.info("Translation Matrix\n%s", t.round(4))

            # Triangulate points
            P1 = np.hstack((self.cameraMat[cam], np.zeros((3, 1))))
            P2 = self.cameraMat[cam + 1] @ np.hstack((R, t.T))

            points3d = self._triangulate_and_filter_outliers(
                P1, P2, centroids1, centroids2
            )

            # Calculate scale based on real-world distances
            real_lengths = [self.L_real_AB, self.L_real_BC, self.L_real_CA]
            lamb, L_vec = self._compute_scale(
                points3d,
                real_lengths,
                method="median",
            )
            points3d_scaled = points3d * lamb

            # Remove outliers based on distance from expected lengths
            filtered_idx, outlier_count = self._remove_distance_outliers(
                points3d_scaled, real_lengths
            )

            if filtered_idx is None or len(filtered_idx) == 0:
                log.warning(
                    f"No valid points left after outlier rejection for CAM{cam}-{cam+1}, skipping refinement."
                )
                continue

            log.info("Refining fundamental matrix estimation using filtered inliers")

            centroids1 = centroids1[filtered_idx]
            centroids2 = centroids2[filtered_idx]

            F, R, t = self._compute_fundamental_essential_and_pose(
                cam, cam + 1, centroids1, centroids2
            )

            if np.isnan(F).any() or np.isnan(R).any() or np.isnan(t).any():
                log.error(
                    f"Refined matrices invalid for CAM{cam}-{cam+1}, skipping update."
                )
                continue

            # Triangulate points again with refined centroids
            P1 = np.hstack((self.cameraMat[cam], np.zeros((3, 1))))
            P2 = self.cameraMat[cam + 1] @ np.hstack((R, t.T))
            points3d = self._triangulate_and_filter_outliers(
                P1, P2, centroids1, centroids2
            )

            # Compute scale again with refined points
            lamb, L_vec = self._compute_scale(points3d, real_lengths, method="mode")
            points3d_scaled = points3d * lamb

            # Log statistics
            self._log_scale_statistics(L_vec, real_lengths, lamb)

            # Save results
            self.calibration_result.rotations.append(R)
            self.calibration_result.translations.append(t)
            self.calibration_result.scales.append([lamb])
            self.calibration_result.fundamental_matrices.append(F)
            self.calibration_result.triangulated_points.append(points3d)

        # Compute projection matrices and transform all points to camera 0 coordinate system
        self._compute_projection_matrices()

    def _find_best_permutation(self, cam, centroids):
        all_perms = list(permutations([0, 1, 2]))
        best_error = float("inf")
        best_perm = None
        best_centroids = centroids

        for perm in all_perms:
            # Permute centroids according to the current permutation
            centroids_perm = np.empty_like(centroids)
            for i, idx in enumerate(perm):
                centroids_perm[i::3, :] = centroids[idx::3, :]

            # Compute measured lengths for the permuted centroids
            meas_lengths = [
                np.linalg.norm(centroids_perm[0] - centroids_perm[1]),  # AB
                np.linalg.norm(centroids_perm[1] - centroids_perm[2]),  # BC
                np.linalg.norm(centroids_perm[2] - centroids_perm[0]),  # CA
            ]

            # Compute the obtained ratios
            meas_ratios = [
                meas_lengths[0] / meas_lengths[1],  # AB/BC
                meas_lengths[1] / meas_lengths[2],  # BC/CA
                meas_lengths[2] / meas_lengths[0],  # CA/AB
            ]

            # Compute the real ratios based on real lengths
            real_ratios = [
                self.expected_ratios[("AB", "BC")],  # AB/BC
                self.expected_ratios[("BC", "CA")],  # BC/CA
                self.expected_ratios[("CA", "AB")],  # CA/AB
            ]

            # Calculate the total error as the sum of absolute differences
            total_error = sum(
                abs(meas_ratio - real_ratio)
                for meas_ratio, real_ratio in zip(meas_ratios, real_ratios)
            )

            log.debug(
                f" Permutation {perm}: measured lengths=[{meas_lengths[0]:.2f}, {meas_lengths[1]:.2f}, {meas_lengths[2]:.2f}], "
            )
            log.debug(
                f"                        measured ratios =[{meas_ratios[0]:.3f}, {meas_ratios[1]:.3f}, {meas_ratios[2]:.3f}], "
            )
            log.debug(
                f"                        real ratios     =[{real_ratios[0]:.3f}, {real_ratios[1]:.3f}, {real_ratios[2]:.3f}], "
            )

            # Update best permutation if the error is lower
            if total_error < best_error:
                best_error = total_error
                best_perm = perm
                # best_centroids = centroids_perm

        log.debug(f" Using permutation {best_perm} for cameras {cam}")
        log.debug("")
        return best_centroids if best_perm is not None else centroids

    def _get_valid_intersections(self, state1, state2):
        return [
            [max(s1, s2), min(e1, e2)]
            for s1, e1 in state1.time_intervals
            for s2, e2 in state2.time_intervals
            if max(s1, s2) <= min(e1, e2)
        ]

    def _interpolate_centroids(self, cam, state1, state2, intersections, verbose=False):
        # Create and fill interpolation dataset
        df_interp = np.zeros((self.nImages, 13))
        df_interp[:, -1] = np.arange(0, self.record, self.step)

        log.debug(
            f"[CAM{cam}-{cam+1}] Interpolation buffer created with shape {df_interp.shape}"
        )
        log.debug(
            f"[CAM{cam}-{cam+1}] Interpolation completed. Non-zero rows: {np.count_nonzero(np.all(df_interp[:, 0:12] != 0, axis=1))}"
        )

        for beg, end in intersections:
            for i, state in enumerate([state1, state2]):
                valid = [
                    j
                    for j, row in enumerate(state.undistorted_frames)
                    if beg <= row[6] <= end
                ]
                if len(valid) <= 2:
                    continue

                coords = state.undistorted_frames[valid, 0:6]
                times = state.undistorted_frames[valid, 6] / 1e6
                t_low, t_high = math.ceil(times[0] / self.step), math.floor(
                    times[-1] / self.step
                )

                if self.verbose:
                    log.info(
                        f"interpolated #{i + cam} from {t_low * self.step:.2f}s to {t_high * self.step:.2f}s"
                    )

                t_new = np.linspace(
                    t_low, t_high, int(t_high - t_low) + 1, dtype=np.uint16
                )
                interp = CubicSpline(times, coords, axis=0)
                df_interp[t_new, i * 6 : i * 6 + 6] = interp(t_new * self.step)

        # Remove rows with zeros
        df_interp = df_interp[np.all(df_interp[:, 0:12] != 0, axis=1)]
        if len(df_interp) < 10:
            log.error(f"No valid overlap between cameras {cam} and {cam+1}")
            return np.nan, np.nan

        centroids1 = df_interp[:, 0:6].reshape(-1, 2)
        centroids2 = df_interp[:, 6:12].reshape(-1, 2)
        log.info(
            f"Interpolated {df_interp.shape[0]} images between cameras {cam} and {cam+1}"
        )
        return centroids1, centroids2

    def _compute_fundamental_essential_and_pose(
        self, cam1, cam2, centroids1, centroids2
    ) -> tuple:
        """
        Computes the fundamental matrix, essential matrix, and camera pose (R, t)
        between two cameras using the provided centroids.

        Parameters:
            cam1 (int): Index of the first camera
            cam2 (int): Index of the second camera
            centroids1 (np.ndarray): 2D points from camera 1
            centroids2 (np.ndarray): 2D points from camera 2

        Returns:
            tuple: Fundamental matrix F, rotation R, translation t
        """
        log.debug(f"Computing F, E, R, t for cameras {cam1} and {cam2}")
        F, _ = estimateFundMatrix_8norm(centroids1, centroids2, verbose=self.verbose)
        if np.any(np.isnan(F)):
            log.error(f"Invalid fundamental matrix for cameras {cam1} and {cam2}")
            return np.nan, np.nan, np.nan

        E = self.cameraMat[cam2].T @ F @ self.cameraMat[cam1]
        R, t = decomposeEssentialMat(
            E,
            self.cameraMat[cam1],
            self.cameraMat[cam2],
            centroids1,
            centroids2,
            cv2_compute=False,
            log=log,
        )

        if np.any(np.isnan(R)) or np.any(np.isnan(t)):
            log.error(
                f"Invalid essential matrix decomposition for cameras {cam1} and {cam2}"
            )
            return np.nan, np.nan, np.nan

        return F, R, t

    def _triangulate_and_filter_outliers(self, P1, P2, centroids1, centroids2) -> tuple:
        """
        Triangulates points from two camera views and filters outliers based on distances.

        Parameters:
            P1 (np.ndarray): Projection matrix for camera 1
            P2 (np.ndarray): Projection matrix for camera 2
            centroids1 (np.ndarray): 2D points from camera 1
            centroids2 (np.ndarray): 2D points from camera 2

        Returns:
            tuple: Triangulated 3D points
        """
        log.debug("Triangulating points...")
        points3d_homogeneous = triangulatePoints(
            P1, P2, projectionPoints(centroids1), projectionPoints(centroids2)
        )
        points3d = (points3d_homogeneous[:3] / points3d_homogeneous[3]).T

        if points3d[0, 2] < 0:
            points3d = -points3d

        # Return triangulated points
        return points3d

    def _compute_scale(self, points3d, real_lengths, method="mode") -> float:
        """
        Computes the scale factor for 3D points based on triplet scales.
        Parameters:
            real_lengths (list): List of real-world distances for triplets
            method (str): Method for scale estimation ("median", "mean", "ransac", "mode")
        Returns:
            float: Computed scale factor
        """
        log.debug("Computing scale factor for 3D points")
        if not isinstance(real_lengths, list) or len(real_lengths) != 3:
            raise ValueError(
                "real_lengths must be a list of three real-world distances for triplets"
            )
        if method not in ["median", "mean", "ransac", "mode"]:
            raise ValueError(
                "method must be one of 'median', 'mean', 'ransac', or 'mode'"
            )
        log.debug(f"Using method: {method} for scale estimation")

        # Step 1: Initialization
        total_scale = 0.0
        L_CA_vec, L_BC_vec, L_AB_vec = [], [], []

        # Step 2: Collect per-triplet scale estimates
        triplet_scales = []
        valid_triplets = []

        for [A, B, C] in points3d.reshape(-1, 3, 3):
            L_rec_CA = np.linalg.norm(C - A)
            L_rec_BC = np.linalg.norm(B - C)
            L_rec_AB = np.linalg.norm(A - B)

            if min(L_rec_CA, L_rec_BC, L_rec_AB) < 1e-6:
                continue  # Avoid degenerate triplets

            scale_CA = real_lengths[0] / L_rec_CA
            scale_BC = real_lengths[1] / L_rec_BC
            scale_AB = real_lengths[2] / L_rec_AB
            scale_mean = np.mean([scale_CA, scale_BC, scale_AB])

            triplet_scales.append(scale_mean)
            valid_triplets.append((A, B, C))
            L_CA_vec.append(L_rec_CA)
            L_BC_vec.append(L_rec_BC)
            L_AB_vec.append(L_rec_AB)

        # Step 3: Filtering using median, mean, or RANSAC median
        triplet_scales = np.array(triplet_scales)
        if method == "ransac":
            threshold = 0.15  # 15% tolerance
            median_scale = np.median(triplet_scales)
            inlier_mask = (
                np.abs(triplet_scales - median_scale) / median_scale < threshold
            )
        else:
            inlier_mask = np.ones_like(triplet_scales, dtype=bool)
        log.debug(
            f"Filtering triplet scales using {method}: obtained {np.sum(inlier_mask)} inliers out of {len(triplet_scales)} total triplets"
        )
        inlier_scales = triplet_scales[inlier_mask]

        if len(inlier_scales) == 0:
            log.warning(
                f"No valid triplets found for scale estimation using {method}, using lamb = 1.0"
            )
            return 1.0

        if method == "median":
            lamb = np.median(inlier_scales)
        elif method == "mean" or method == "ransac":
            lamb = np.mean(inlier_scales)
        elif method == "mode":
            from scipy import stats

            lamb = stats.mode(inlier_scales)[0][0]

        log.debug(f"Computed scale factor: {lamb:.4f}")
        return lamb, {
            "L_AB": np.array(L_AB_vec) * lamb,
            "L_BC": np.array(L_BC_vec) * lamb,
            "L_CA": np.array(L_CA_vec) * lamb,
        }

    def _remove_distance_outliers(self, points3d, real_lengths) -> tuple:
        """
        Removes outliers based on distance from the origin in the 3D point cloud.

        Parameters:
            points3d (np.ndarray): 3D points to filter
            real_lengths (list): List of real-world distances for triplets

        Returns:
            tuple: Indices of valid points and count of outliers removed
        """
        log.debug("Removing distance outliers from 3D points")
        if len(points3d) < 3:
            log.warning("Not enough points to remove outliers, returning all points")
            return np.arange(points3d.shape[0]), 0
        # Calculate distances for each triplet
        distances = {
            "L_AB": np.linalg.norm(points3d[::3] - points3d[1::3], axis=1),
            "L_BC": np.linalg.norm(points3d[1::3] - points3d[2::3], axis=1),
            "L_CA": np.linalg.norm(points3d[2::3] - points3d[::3], axis=1),
        }
        log.debug(
            f" Calculated Distances: L_AB[0]= {distances['L_AB'][0]:.3f}, L_BC[0]= {distances['L_BC'][0]:.3f}, L_CA[0]= {distances['L_CA'][0]:.3f}"
        )
        # Calculate the mean and standard deviation for each distance
        means = {key: np.mean(val) for key, val in distances.items()}
        stds = {key: np.std(val) for key, val in distances.items()}
        log.debug(
            f" Means: {{L_AB: {means['L_AB']:.3f}, L_BC: {means['L_BC']:.3f}, L_CA: {means['L_CA']:.3f}}}"
        )
        log.debug(
            f" Stds : {{L_AB: {stds['L_AB']:.3f}, L_BC: {stds['L_BC']:.3f}, L_CA: {stds['L_CA']:.3f}}}"
        )
        # Calculate the threshold for outliers
        outlier_threshold_method = "mean-std"  # Options: "mean-std", "real"
        if outlier_threshold_method == "mean-std":
            # Use mean + 1.5 * std for outlier detection
            log.debug(" Using mean + 1.5 * std for outlier thresholds")
            thresholds = {
                key: (means[key] - 1.5 * stds[key], means[key] + 1.5 * stds[key])
                for key in distances
            }
        elif outlier_threshold_method == "real":
            # Use real-world lengths with tolerance
            log.debug(" Using real-world lengths for outlier thresholds")
            if not all(isinstance(length, (int, float)) for length in real_lengths):
                raise ValueError("real_lengths must contain numeric values")
            if len(real_lengths) != 3:
                raise ValueError("real_lengths must contain exactly three values")
            log.debug(f" Real lengths: {real_lengths}")
            # Calculate thresholds based on real lengths and outlier tolerance
            self.outlier_tolerance = 0.40
            log.debug(f" Using outlier tolerance: {self.outlier_tolerance}")
            thresholds = {
                "L_AB": (
                    real_lengths[0] * (1 - self.outlier_tolerance),
                    real_lengths[0] * (1 + self.outlier_tolerance),
                ),
                "L_BC": (
                    real_lengths[1] * (1 - self.outlier_tolerance),
                    real_lengths[1] * (1 + self.outlier_tolerance),
                ),
                "L_CA": (
                    real_lengths[2] * (1 - self.outlier_tolerance),
                    real_lengths[2] * (1 + self.outlier_tolerance),
                ),
            }
        log.debug(
            f"Thresholds: {{L_AB: ({thresholds['L_AB'][0]:.3f}, {thresholds['L_AB'][1]:.3f}), "
            f"L_BC: ({thresholds['L_BC'][0]:.3f}, {thresholds['L_BC'][1]:.3f}), "
            f"L_CA: ({thresholds['L_CA'][0]:.3f}, {thresholds['L_CA'][1]:.3f})}}"
        )

        # Identify outliers based on the thresholds
        valid_indices = []
        outlier_count = 0
        for i in range(0, len(points3d), 3):
            A, B, C = points3d[i : i + 3]
            L_AB = np.linalg.norm(A - B)
            L_BC = np.linalg.norm(B - C)
            L_CA = np.linalg.norm(C - A)

            if (
                thresholds["L_AB"][0] <= L_AB <= thresholds["L_AB"][1]
                and thresholds["L_BC"][0] <= L_BC <= thresholds["L_BC"][1]
                and thresholds["L_CA"][0] <= L_CA <= thresholds["L_CA"][1]
            ):
                valid_indices.extend([i, i + 1, i + 2])
            else:
                outlier_count += 1

        log.info(
            f"Outliers removed: {outlier_count} out of {len(points3d) // 3} triplets"
        )
        # Return valid indices and count of outliers
        return np.array(valid_indices), outlier_count

    def _log_scale_statistics(self, L_vec, real_lengths, lamb) -> None:
        """
        Logs the scale statistics for the triangulated points.

        Parameters:
            L_vec (dict): Dictionary containing scaled lengths for triplets
            real_lengths (list): List of real-world distances for triplets
            lamb (float): Computed scale factor
        """
        key_to_index = {"L_AB": 0, "L_BC": 1, "L_CA": 2}

        log.info("")
        log.info(
            f"Scale between real world and triangulated point cloud is: {lamb:.2f}"
        )
        for key, value in L_vec.items():
            idx = key_to_index.get(key, None)
            if idx is None:
                log.warning(f"Unknown length key '{key}', skipping.")
                continue
            log.info(
                f"  {key}: mean={np.mean(value):.2f}, std={np.std(value):.2f}, "
                f"real={real_lengths[idx]:.2f} cm"
            )

    def _compute_projection_matrices(self) -> None:
        """
        Compute projection matrices for all cameras relative to camera 0
        and transform all triangulated points to the same coordinate system.
        """
        proj_matrices, all_points_3d = [], []

        for cam in range(self.cameras):
            # Initialize projection matrix as identity
            P_new = np.vstack(
                (
                    np.hstack((np.identity(3), np.zeros((3, 1)))),
                    np.hstack((np.zeros(3), 1)),
                )
            )

            # Chain transformations from current camera back to camera 0
            for i in reversed(range(cam)):
                if i < len(self.calibration_result.translations):
                    t = np.array(self.calibration_result.translations[i][0]).reshape(
                        -1, 3
                    )
                    R = np.array(self.calibration_result.rotations[i])
                    lamb = self.calibration_result.scales[i][0]
                    t_new = np.matmul(-t, R).reshape(-1, 3) * lamb / 100
                    P = np.vstack(
                        (np.hstack((R.T, t_new.T)), np.hstack((np.zeros(3), 1)))
                    )
                    P_new = np.matmul(P, P_new)

            proj_matrices.append(P_new)

            # Transform triangulated points to camera 0 coordinate system
            if cam < len(self.calibration_result.triangulated_points):
                points3d = self.calibration_result.triangulated_points[cam]
                scale_factor = (
                    self.calibration_result.scales[cam + 1][0]
                    if cam + 1 < len(self.calibration_result.scales)
                    else 1.0
                )
                points_homogeneous = np.hstack(
                    (points3d * scale_factor / 100, np.ones((points3d.shape[0], 1)))
                ).T
                transformed_points = np.matmul(P_new, points_homogeneous)
                all_points_3d.append(transformed_points)

        self.calibration_result.projection_matrices = proj_matrices
        self.calibration_result.all_points_3d = (
            np.hstack(all_points_3d) if all_points_3d else np.zeros((4, 0))
        )

    def _save_calibration_results(self) -> None:
        """
        Save calibration results to CSV files and display interactive 3D plot.
        """
        result = self.calibration_result

        # Save matrices to CSV files
        np.savetxt(
            "mcr/capture/data/R.csv", np.array(result.rotations).ravel(), delimiter=","
        )
        np.savetxt(
            "mcr/capture/data/t.csv",
            np.array(result.translations).ravel(),
            delimiter=",",
        )
        np.savetxt(
            "mcr/capture/data/lamb.csv", np.array(result.scales).ravel(), delimiter=","
        )
        np.savetxt(
            "mcr/capture/data/F.csv",
            np.array(result.fundamental_matrices).ravel(),
            delimiter=",",
        )
        np.savetxt(
            "mcr/capture/data/projMat.csv",
            np.array(result.projection_matrices).ravel(),
            delimiter=",",
        )
        np.savetxt(
            "mcr/capture/data/all_points.csv",
            result.all_points_3d[:3].T,  # one point per row
            fmt="%.6f",
            delimiter=",",
        )

        # Start interactive ArenaViewer
        viewer = ArenaViewer(title="3D Map of the Calibration Process", arenaSize=2)

        # Add cameras as frames
        for i, P in enumerate(result.projection_matrices):
            log.debug(f"Adding camera {i} frame to viewer")
            log.debug(f"Projection Matrix P{i}:\n{P}")
            R = P[:3, :3]
            t = (P @ np.array([[0], [0], [0], [1]])).reshape(-1, 1)[:3]
            viewer.add_frame(Frame(R=R, t=t), name=f"Camera {i}")

        # Add all 3D triangulated points
        viewer.add_markers(
            result.all_points_3d[:3], name="Triangulated Points", color="darkred"
        )

        log.info(f"Total triangulated points: {result.all_points_3d.shape[1]}")
        log.info("Sample triangulated 3D points (x, y, z):")
        for i in range(min(15, result.all_points_3d.shape[1])):  # Log first 15
            x, y, z = (
                result.all_points_3d[0, i],
                result.all_points_3d[1, i],
                result.all_points_3d[2, i],
            )
            log.info(f"  Point {i+1}: ({x:.4f}, {y:.4f}, {z:.4f})")

        viewer.figure.show()

    def get_results(self) -> CalibrationResult:
        """
        Return the calibration results.

        Returns:
            CalibrationResult: Complete calibration data including matrices and points
        """
        return self.calibration_result
