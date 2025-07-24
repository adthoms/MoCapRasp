import numpy as np

# === Camera intrinsics and distortion coefficients for each camera ===
# Format: [fx, 0, cx], [0, fy, cy], [0, 0, 1]
# Distortion: [k1, k2, p1, p2]

_fx = [763.37, 748.25, 752.23]
_fy = [762.67, 750.19, 754.14]
_cx = [483.41, 471.80, 486.83]
_cy = [330.11, 299.70, 311.30]

_intrinsics = [
    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    for fx, fy, cx, cy in zip(_fx, _fy, _cx, _cy)
]

_distortions = [
    [0.1748, -0.3099, -0.0001, 0.0024],
    [0.1690, -0.2976, -0.0010, 0.0024],
    [0.1670, -0.2836, -0.0028, 0.0007],
]

# Construct cameraMat and distCoef
cameraMat = [np.array(K, dtype=np.float64) for K in _intrinsics]
distCoef = [np.array(D, dtype=np.float32).reshape(-1, 1) for D in _distortions]

if __name__ == "__main__":
    for i, (K, D) in enumerate(zip(cameraMat, distCoef)):
        print(f"Camera {i}:")
        print("  Intrinsic Matrix:\n", K)
        print("  Distortion Coefficients:\n", D)