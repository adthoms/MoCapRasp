import os, math
import warnings
import logging
import numpy as np
from datetime import datetime
from collections import Counter
from itertools import combinations
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, field
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
from mcr.misc.plot import ArenaViewer, Frame


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class CameraState:
    """
    Holds per-camera state during capture, including frame counters, timestamps,
    undistorted marker coordinates, and certainty intervals for calibration.
    """

    capture_active: bool = True  # Whether this camera is still streaming
    frame_counter: int = 0  # Number of frames successfully received
    last_timestamp: int = 0  # Timestamp of the last valid frame
    missed_frames: int = 0  # Count of missed frames due to parsing errors or occlusion
    invalid_frames: int = 0  # Count of invalid frames since last good frame
    swap_counter: int = 0  # Counter for marker reordering validation
    has_certainty: bool = False  # Whether the current marker sequence is confirmed
    last_image_id: int = -1  # Last received image ID from this camera
    intervals: list = field(
        default_factory=list
    )  # Frame-based indices where marker certainty begins
    time_intervals: list = field(
        default_factory=list
    )  # List of valid timestamp intervals for calibration
    undistorted_frames: list = field(
        default_factory=list
    )  # All undistorted marker coordinates with timestamps


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
    translations: list = field(default_factory=list)  # List of relative translations
    scales: list = field(
        default_factory=list
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
                    log.info(
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
                f"Camera {idx}: certainty={state.has_certainty}, intervals={len(state.time_intervals)}"
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
                log.info(f"CAM{i} intervals: {state.time_intervals}")

        self._compute_camera_extrinsics_from_intervals()

        if self.save:
            self._save_calibration_results()

    def _compute_camera_extrinsics_from_intervals(self) -> None:
        """
        Performs pairwise camera extrinsics calibration for all consecutive camera pairs,
        estimates F and E, decomposes into R and t, triangulates, computes scale,
        and stores projection matrices and triangulated points.
        """
        # Define true distances from your marker configuration
        L_AC = np.linalg.norm(np.array([0, 0]) - np.array([0, 10.4]))  # Marker 0 to 3
        L_AB = np.linalg.norm(np.array([0, 0]) - np.array([9, 0]))  # Marker 0 to 1
        L_BC = np.linalg.norm(np.array([9, 0]) - np.array([0, 6.1]))  # Marker 1 to 2

        for cam in range(self.cameras - 1):
            state1, state2 = self.camera_states[cam], self.camera_states[cam + 1]

            intersections = [
                [max(s1, s2), min(e1, e2)]
                for s1, e1 in state1.time_intervals
                for s2, e2 in state2.time_intervals
                if max(s1, s2) <= min(e1, e2)
            ]
            if self.verbose:
                log.info(f"Intersection of CAM{cam} and CAM{cam+1}: {intersections}")

            df_interp = np.zeros((self.nImages, 13))
            df_interp[:, -1] = np.arange(0, self.record, self.step)

            for beg, end in intersections:
                for i, state in enumerate([state1, state2]):
                    valid = [
                        j
                        for j, row in enumerate(state.undistorted_frames)
                        if beg <= row[6] <= end
                    ]
                    if len(valid) < 3:
                        continue
                    coords = state.undistorted_frames[valid, 0:6]
                    times = state.undistorted_frames[valid, 6] / 1e6
                    t_low, t_high = math.ceil(times[0] / self.step), math.floor(
                        times[-1] / self.step
                    )
                    t_new = np.linspace(
                        t_low, t_high, int(t_high - t_low) + 1, dtype=np.uint16
                    )
                    interp = CubicSpline(times, coords, axis=0)
                    df_interp[t_new, i * 6 : i * 6 + 6] = interp(t_new * self.step)

            df_interp = df_interp[np.all(df_interp[:, 0:12] != 0, axis=1)]
            if len(df_interp) == 0:
                log.error(f"[ERROR] No valid overlap between cameras {cam} and {cam+1}")
                continue

            c1, c2 = df_interp[:, 0:6].reshape(-1, 2), df_interp[:, 6:12].reshape(-1, 2)
            F, _ = estimateFundMatrix_8norm(c1, c2, verbose=self.verbose)
            if np.any(np.isnan(F)) or np.linalg.matrix_rank(F) < 2:
                log.error(f"[ERROR] Invalid F matrix for CAM{cam}-{cam+1}")
                continue

            E = self.cameraMat[cam + 1].T @ F @ self.cameraMat[cam]
            if np.any(np.isnan(E)) or np.linalg.matrix_rank(E) < 2:
                log.error(f"[ERROR] Invalid E matrix for CAM{cam}-{cam+1}")
                continue

            R, t = decomposeEssentialMat(
                E,
                self.cameraMat[cam],
                self.cameraMat[cam + 1],
                c1,
                c2,
                cv2_compute=True,
            )
            if (
                np.any(np.isnan(R))
                or np.linalg.matrix_rank(R) < 3
                or np.linalg.norm(t) == 0
            ):
                log.error(f"[ERROR] Invalid R or t for CAM{cam}-{cam+1}")
                continue

            P1 = np.hstack((self.cameraMat[cam], np.zeros((3, 1))))
            P2 = self.cameraMat[cam + 1] @ np.hstack((R, t.reshape(3, 1)))
            points4d = triangulatePoints(
                P1,
                P2,
                projectionPoints(c1),
                projectionPoints(c2),
            )
            points3d = (points4d[:3] / points4d[3]).T
            if np.mean(points3d[:, 2]) < 0:
                points3d = -points3d

            log.info(f"Triangulated 3D points shape: {points3d.shape}")
            log.info(f"Sample triangulated points (first 3):\n{points3d[:3]}")

            # Compute scale
            ratios = []
            for A, B, C in points3d.reshape(-1, 3, 3):
                try:
                    d_ac = np.linalg.norm(A - C)
                    d_ab = np.linalg.norm(A - B)
                    d_bc = np.linalg.norm(B - C)

                    # Skip degenerate cases to avoid division by zero
                    if d_ac < 1e-6 or d_ab < 1e-6 or d_bc < 1e-6:
                        continue

                    ratios.extend(
                        [
                            L_AC / d_ac,
                            L_AB / d_ab,
                            L_BC / d_bc,
                        ]
                    )
                except Exception as e:
                    log.warning(f"Skipping a triplet due to error: {e}")
                    continue

            if len(ratios) == 0:
                log.error("No valid scale ratios found. Cannot compute lamb.")
                lamb = 1.0  # fallback value
            else:
                lamb = np.median(ratios)

            log.info(f"Computed scale lamb: {lamb:.2f}")
            points3d_scaled = points3d * lamb

            # Consensus points
            consensus_points = self.compute_consensus_points(
                points3d_scaled,
                n=4,
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
            )
            if len(consensus_points) < 4:
                log.warning(
                    f"[CAM{cam}-{cam+1}] Not enough consensus points found. Using all points."
                )
                consensus_points = points3d_scaled
            else:
                consensus_points = np.array(consensus_points)
                consensus_points = np.unique(consensus_points, axis=0)
                consensus_points = consensus_points[
                    np.linalg.norm(consensus_points, axis=1) < 100
                ]

            log.info(f"[CONSENSUS] Found {len(consensus_points)} stable 3D points:")
            for i, pt in enumerate(consensus_points):
                log.info(f"  Point {i+1}: ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")

            self.calibration_result.rotations.append(R)
            self.calibration_result.translations.append(t.reshape(-1).tolist())
            self.calibration_result.scales.append(float(lamb))
            self.calibration_result.triangulated_points.append(points3d_scaled)

            log.info(
                f"[CAM{cam}-{cam+1}] Calibration done. Scale: {lamb:.2f}, Points: {len(points3d_scaled)}"
            )

        base_lambda = self.calibration_result.scales[0]
        self.calibration_result.scales = [
            1.0 if i == 0 else s / base_lambda
            for i, s in enumerate(self.calibration_result.scales)
        ]

        # Compute projection matrices
        proj_matrices, all_points_3d = [], []
        for cam in range(self.cameras):
            P_cam = np.eye(4)
            for i in reversed(range(cam)):
                R = self.calibration_result.rotations[i]
                t = np.array(self.calibration_result.translations[i]).reshape(1, 3)
                lamb = self.calibration_result.scales[i]
                # t_adj = (-t @ R).reshape(-1, 1) * lamb / 100
                # T = np.vstack((np.hstack((R.T, t_adj)), [0, 0, 0, 1]))
                T = np.eye(4)
                T[:3, :3] = R.T
                T[:3, 3] = -R.T @ (t.flatten() * lamb / 100)
                P_cam = T @ P_cam
            proj_matrices.append(P_cam)

            if cam < len(self.calibration_result.triangulated_points):
                pts3d = self.calibration_result.triangulated_points[cam]
                lamb_next = (
                    self.calibration_result.scales[cam + 1]
                    if cam + 1 < len(self.calibration_result.scales)
                    else 1.0
                )
                pts_h = np.hstack(
                    (pts3d * lamb_next / 100, np.ones((pts3d.shape[0], 1)))
                ).T
                transformed = (P_cam @ pts_h).T
                all_points_3d.append(transformed)

        self.calibration_result.projection_matrices = proj_matrices
        self.calibration_result.all_points_3d = (
            np.vstack(all_points_3d).T if all_points_3d else np.zeros((4, 0))
        )

        for i, P in enumerate(self.calibration_result.projection_matrices):
            pos = (P @ np.array([[0], [0], [0], [1]])).ravel()[:3]
            log.info(f"Camera {i} world position: {pos}")

    def _save_calibration_results(self) -> None:
        result = self.calibration_result
        np.savetxt("mcr/capture/data/R.csv", np.ravel(result.rotations), delimiter=",")
        np.savetxt(
            "mcr/capture/data/t.csv", np.ravel(result.translations), delimiter=","
        )
        np.savetxt("mcr/capture/data/lamb.csv", np.ravel(result.scales), delimiter=",")
        np.savetxt(
            "mcr/capture/data/F.csv",
            np.ravel(result.fundamental_matrices),
            delimiter=",",
        )
        np.savetxt(
            "mcr/capture/data/projMat.csv",
            np.ravel(result.projection_matrices),
            delimiter=",",
        )

        all_points = result.all_points_3d

        # Start interactive ArenaViewer
        viewer = ArenaViewer(title="Camera Calibration Results", arenaSize=2)

        # Add cameras as frames
        for i, P in enumerate(result.projection_matrices):
            R = P[:3, :3]
            t = (P @ np.array([[0], [0], [0], [1]])).reshape(-1, 1)[:3]
            viewer.add_frame(Frame(R=R, t=t), name=f"Camera {i}")

        # Add all 3D triangulated points
        viewer.add_markers(all_points[:3], name="Triangulated Points", color="darkred")

        log.info(f"Total triangulated points: {all_points.shape[1]}")
        log.info("Sample triangulated 3D points (x, y, z):")
        for i in range(min(15, all_points.shape[1])):  # Log first 15
            x, y, z = all_points[0, i], all_points[1, i], all_points[2, i]
            log.info(f"  Point {i+1}: ({x:.4f}, {y:.4f}, {z:.4f})")

        viewer.figure.show()

    def compute_consensus_points(self, points3d, n=4, eps=0.01, min_samples=10):
        """
        Cluster 3D points and return centroids of the n most populated clusters.
        """
        if points3d.shape[0] < min_samples:
            log.warning("Not enough points for DBSCAN clustering.")
            return points3d

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points3d)
        labels = clustering.labels_
        label_counts = Counter(labels[labels != -1])  # ignore noise

        if not label_counts:
            log.warning("No clusters found by DBSCAN.")
            return points3d

        top_labels = [label for label, _ in label_counts.most_common(n)]

        consensus_points = []
        for label in top_labels:
            pts = points3d[labels == label]
            if pts.size == 0:
                continue
            centroid = np.mean(pts, axis=0)
            consensus_points.append(centroid)

        return np.array(consensus_points)

    def get_results(self) -> CalibrationResult:
        return self.calibration_result
