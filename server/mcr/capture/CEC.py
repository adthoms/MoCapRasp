import os, math
import warnings
import logging
import numpy as np
from datetime import datetime
from itertools import combinations
from scipy.interpolate import CubicSpline
from cv2 import destroyAllWindows, triangulatePoints

from mcr.capture.CaptureProcess import CaptureProcess
from mcr.misc.math import isCollinear
from mcr.misc.cameras import (
    estimateFundMatrix_8norm,
    decomposeEssentialMat,
    projectionPoints,
)
from mcr.misc.markers import orderCenterCoord, occlusion, processCentroids
from mcr.misc.plot import plotArena


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class CameraState:
    """
    Holds per-camera state during capture, including frame counters, timestamps,
    undistorted marker coordinates, and certainty intervals for calibration.
    """

    def __init__(self) -> None:
        self.capture_active = True  # Whether this camera is still streaming
        self.frame_counter = 0  # Number of frames successfully received
        self.last_timestamp = 0  # Timestamp of the last valid frame
        self.missed_frames = (
            0  # Count of missed frames due to parsing errors or occlusion
        )
        self.invalid_frames = 0  # Count of invalid frames since last good frame
        self.swap_counter = 0  # Counter for marker reordering validation
        self.has_certainty = False  # Whether the current marker sequence is confirmed
        self.last_image_id = -1  # Last received image ID from this camera
        self.intervals = []  # Frame-based indices where marker certainty begins
        self.time_intervals = []  # List of valid timestamp intervals for calibration
        self.undistorted_frames = (
            []
        )  # All undistorted marker coordinates with timestamps


class CalibrationResult:
    """
    Holds the full output of the multi-camera calibration process.
    Includes rotation, translation, scaling factors, triangulated points,
    and projection matrices.
    """

    def __init__(self) -> None:
        self.rotations = [
            np.identity(3)
        ]  # List of relative rotation matrices between camera pairs
        self.translations = [[[0.0, 0.0, 0.0]]]  # List of relative translations
        self.scales = [[1]]  # List of scale factors for 3D point normalization
        self.fundamental_matrices = []  # Fundamental matrices (2D epipolar geometry)
        self.triangulated_points = (
            []
        )  # 3D points after triangulation and scale application
        self.all_points_3d = []  # Reserved for storing unified 3D data (not yet used)
        self.projection_matrices = []  # Reserved for projection matrices (optional use)


class CEC(CaptureProcess):
    """
    Camera Extrinsics Calibration (CEC) process.

    Inherits from CaptureProcess and implements a multi-camera calibration
    routine including time-synchronized capture, marker ordering, 2D interpolation,
    essential matrix decomposition, triangulation, and export of camera extrinsics.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.camera_states = [CameraState() for _ in range(self.cameras)]
        self.calibration_result = CalibrationResult()

    def collect(self) -> None:
        """
        Main loop to receive data packets from all cameras, process and undistort the
        detected marker blobs, track certainty, and accumulate all valid 2D points.
        When capture is complete, invokes the calibration procedure.
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
            self._finalize_and_calibrate(saved_data_rows)

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
        if not (size_msg - 1):
            cam_state.capture_active = False
            return

        if size_msg < 13:
            if self.verbose:
                log.error(f"[CAM{idx}] Only {(size_msg - 4) // 3} markers were found")
            cam_state.missed_frames += 1
            return

        msg = message[0 : size_msg - 4].reshape(-1, 3)
        coord, size = msg[:, 0:2], msg[:, 2].reshape(-1)

        if size_msg > 13:
            order = np.argsort(size)[::-1]
            coord_sorted = coord[order[:4]]
            max_area, best_triplet = -1, None
            for combo in combinations(coord_sorted, 3):
                a, b, c = combo
                area = 0.5 * abs(np.cross(b - a, c - a))
                if area > max_area:
                    max_area = area
                    best_triplet = np.array([a, b, c])
            coord = best_triplet
            if self.verbose:
                log.debug(f"[CAM{idx}] Selected blobs with area: {max_area:.2f}")
            if max_area < 10:
                if self.verbose:
                    log.warning(f"[CAM{idx}] Triangle too small: area = {max_area:.2f}")
                cam_state.missed_frames += 1
                return

        a, b, timestamp, img_number = (
            message[-4],
            message[-3],
            message[-2],
            int(message[-1]),
        )
        und_coord = processCentroids(
            coord, a, b, self.cameraMat[idx], self.distCoef[idx]
        )

        if self.save:
            saved_data_rows.append(
                np.concatenate((und_coord.reshape(6), [timestamp, img_number, idx]))
            )

        if cam_state.frame_counter:
            if abs(timestamp - cam_state.last_timestamp) > 1e9:
                if self.verbose:
                    log.warning(f"[CAM{idx}] Time mismatch")
                cam_state.missed_frames += 1
                cam_state.invalid_frames += 1
                return

        if img_number > cam_state.last_image_id + 1:
            cam_state.invalid_frames = img_number - cam_state.last_image_id

        if not occlusion(und_coord, 5) and not np.any(und_coord < 0):
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
                prev = (
                    np.array(cam_state.undistorted_frames[-1][0:6]).reshape(-1, 2)
                    if cam_state.frame_counter > 0
                    else np.array(cam_state.undistorted_frames[0:6]).reshape(-1, 2)
                )
            und_coord, _ = orderCenterCoord(und_coord, prev)
            und_coord = np.array(und_coord)
        else:
            if self.verbose:
                log.warning(f"[CAM{idx}] Occluded or invalid coordinates")
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

        if not cam_state.has_certainty:
            for [A, B, C] in und_coord.reshape([-1, 3, 2]):
                ab, cb = np.linalg.norm(A - B), np.linalg.norm(C - B)
                if self.verbose:
                    log.debug(
                        f"[CAM{idx}] Ratios: AB/CB = {ab/cb:.2f}, CB/AB = {cb/ab:.2f}, AB = {ab:.2f}"
                    )
                if max_area > 5000 and ab > 20:
                    cam_state.swap_counter += 1
                    if cam_state.swap_counter > 1:
                        cam_state.swap_counter = 0
                        cam_state.has_certainty = True
                        start = cam_state.intervals[-1]
                        end = cam_state.frame_counter
                        (
                            cam_state.undistorted_frames[start:end, 0:2],
                            cam_state.undistorted_frames[start:end, 4:6],
                        ) = np.copy(
                            cam_state.undistorted_frames[start:end, 4:6]
                        ), np.copy(
                            cam_state.undistorted_frames[start:end, 0:2]
                        )
                if cb / ab > (2 - 0.5) and cb > 20:
                    cam_state.has_certainty = True

    def _finalize_and_calibrate(self, saved_data_rows) -> None:
        """
        After all cameras finish streaming, this method saves the undistorted 2D marker data,
        logs camera summaries, and launches the extrinsics calibration step.
        """
        self.server_socket.close()
        destroyAllWindows()

        if self.save:
            now = datetime.now()
            ymd, HMS = now.strftime("%y-%m-%d"), now.strftime("%H-%M-%S")
            path = f"debug/dataSaves/{ymd}/"
            if not os.path.exists(path):
                os.makedirs(path)
                log.info(f"Folder {path} created!")
            np.savetxt(
                path + f"CEC-{HMS}.csv", np.array(saved_data_rows), delimiter=","
            )

        for idx, state in enumerate(self.camera_states):
            log.info(
                f"[SUMMARY] Camera {idx}: certainty={state.has_certainty}, intervals={len(state.time_intervals)}"
            )
            if len(state.undistorted_frames) == 0:
                continue
            if state.has_certainty:
                beg, end = state.intervals[-1], state.frame_counter - 1
                state.time_intervals.append(
                    [state.undistorted_frames[beg, 6], state.undistorted_frames[end, 6]]
                )
                if self.verbose:
                    log.info(
                        f"[CAM{idx}] valid from {state.undistorted_frames[beg, 6] / 1e6:.2f}s to {state.undistorted_frames[end, 6] / 1e6:.2f}s"
                    )

        log.info("[RESULTS] server results are")
        for i, state in enumerate(self.camera_states):
            log.info(
                f"  >> camera {i}: {len(state.undistorted_frames)} valid images, address {self.ipList[i]}, missed {state.missed_frames} images"
            )

        if self.verbose:
            for i, state in enumerate(self.camera_states):
                log.debug(f"[DEBUG] CAM{i} intervals: {state.time_intervals}")

        self._compute_camera_extrinsics_from_intervals()

    def _compute_camera_extrinsics_from_intervals(self) -> None:
        """
        Performs pairwise camera extrinsics calibration for all consecutive camera pairs
        using overlapping valid intervals. It estimates F and E matrices, decomposes E
        into R and t, triangulates 3D points, computes scale, and saves the results
        into compressed .npz bundles per camera pair.
        """
        # Define true distances from your marker configuration
        L_AC = np.linalg.norm(np.array([0, 0]) - np.array([0, 10.4]))  # Marker 0 to 3
        L_AB = np.linalg.norm(np.array([0, 0]) - np.array([9, 0]))  # Marker 0 to 1
        L_BC = np.linalg.norm(np.array([9, 0]) - np.array([0, 6.1]))  # Marker 1 to 2

        for cam in range(self.cameras - 1):
            state1 = self.camera_states[cam]
            state2 = self.camera_states[cam + 1]

            intersections = [
                [max(start1, start2), min(end1, end2)]
                for start1, end1 in state1.time_intervals
                for start2, end2 in state2.time_intervals
                if max(start1, start2) <= min(end1, end2)
            ]

            if self.verbose:
                log.debug(
                    f"[DEBUG] Intersection of CAM{cam} and CAM{cam+1}: {intersections}"
                )

            df_interp = np.zeros((self.nImages, 13))
            df_interp[:, -1] = np.arange(0, self.record, self.step)

            for beg, end in intersections:
                for i, s in enumerate([state1, state2]):
                    valid_idx = [
                        j
                        for j in range(len(s.undistorted_frames))
                        if beg <= s.undistorted_frames[j, 6] <= end
                    ]
                    if len(valid_idx) < 3:
                        continue
                    coords = s.undistorted_frames[valid_idx, 0:6]
                    times = s.undistorted_frames[valid_idx, 6] / 1e6
                    t_low, t_high = math.ceil(times[0] / self.step), math.floor(
                        times[-1] / self.step
                    )
                    t_new = np.linspace(
                        t_low, t_high, int(t_high - t_low) + 1, dtype=np.uint16
                    )
                    interp = CubicSpline(times, coords, axis=0)
                    df_interp[t_new, i * 6 : i * 6 + 6] = interp(t_new * self.step)

            df_interp = df_interp[np.all(df_interp[:, 0:12] != 0, axis=1)]
            if df_interp.shape[0] == 0:
                log.error(f"[ERROR] No valid overlap between cameras {cam} and {cam+1}")
                continue

            centroids1 = df_interp[:, 0:6].reshape(-1, 2)
            centroids2 = df_interp[:, 6:12].reshape(-1, 2)

            F, _ = estimateFundMatrix_8norm(
                centroids1, centroids2, verbose=self.verbose
            )
            if np.any(np.isnan(F)) or np.linalg.matrix_rank(F) < 2:
                log.error(f"[ERROR] Invalid F matrix for CAM{cam}-{cam+1}")
                continue

            E = self.cameraMat[cam + 1].T @ F @ self.cameraMat[cam]
            U, S, Vt = np.linalg.svd(E)
            log.debug(f"[DEBUG] E SVD: singular values = {S}")
            if np.any(np.isnan(E)) or np.linalg.matrix_rank(E) < 2:
                log.error(f"[ERROR] Invalid E matrix for CAM{cam}-{cam+1}")
                continue

            log.debug(f"[DEBUG] Fundamental matrix (F):\n{F}")
            log.debug(f"[DEBUG] Essential matrix (E):\n{E}")

            R, t = decomposeEssentialMat(
                E,
                self.cameraMat[cam],
                self.cameraMat[cam + 1],
                centroids1,
                centroids2,
                cv2_compute=True,
            )
            log.debug(f"[DEBUG] Rotation matrix (R):\n{R}")
            log.debug(f"[DEBUG] Translation vector (t): {t}")

            if np.any(np.isnan(R)) or np.linalg.matrix_rank(R) < 3:
                log.error(f"[ERROR] No valid rotation matrix for CAM{cam}-{cam+1}")
                continue
            if np.linalg.norm(t) == 0 or np.any(np.isnan(t)):
                log.error(f"[ERROR] No valid translation vector for CAM{cam}-{cam+1}")
                continue

            log.debug(f"[DEBUG] Proceeding with triangulation for CAM{cam}-{cam+1}")

            P1 = np.hstack((self.cameraMat[cam], np.zeros((3, 1))))
            P2 = self.cameraMat[cam + 1] @ np.hstack((R, t.reshape(3, 1)))
            projPt1, projPt2 = projectionPoints(centroids1), projectionPoints(
                centroids2
            )
            points4d = triangulatePoints(P1, P2, projPt1, projPt2)
            points3d = (points4d[:3, :] / points4d[3, :]).T

            if points3d[0, 2] < 0:
                points3d = -points3d

            log.debug(f"[DEBUG] Triangulated 3D points shape: {points3d.shape}")
            log.debug(f"[DEBUG] Sample triangulated points (first 3):\n{points3d[:3]}")

            total, count = 0, 0
            for [A, B, C] in points3d.reshape(-1, 3, 3):
                Lr_AC = np.linalg.norm(A - C)
                Lr_AB = np.linalg.norm(A - B)
                Lr_BC = np.linalg.norm(B - C)
                total += L_AC / Lr_AC + L_AB / Lr_AB + L_BC / Lr_BC
                count += 3

            lamb = total / count
            points3d_scaled = points3d * lamb

            log.debug(f"[DEBUG] Computed scale factor (lambda): {lamb}")

            self.calibration_result.rotations.append(R)
            self.calibration_result.translations.append([t.tolist()])
            self.calibration_result.scales.append([lamb])
            self.calibration_result.fundamental_matrices.append(F)
            self.calibration_result.triangulated_points.append(points3d_scaled)

            log.info(
                f"[CAM{cam}-{cam+1}] Calibration complete. "
                f"Scale: {lamb:.2f}, Points: {points3d_scaled.shape[0]}"
            )

            output_path = f"debug/calibration/cam{cam}_cam{cam+1}/"
            os.makedirs(output_path, exist_ok=True)

            np.savez_compressed(
                os.path.join(output_path, "extrinsics.npz"),
                R=R,
                t=t,
                scale=lamb,
                F=F,
                E=E,
                points3d=points3d_scaled,
                K1=self.cameraMat[cam],
                K2=self.cameraMat[cam + 1],
            )
