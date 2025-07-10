import numpy as np

# === Camera intrinsics and distortion coefficients for each camera ===
# Format: [fx, 0, cx], [0, fy, cy], [0, 0, 1]
# Distortion: [k1, k2, p1, p2]

_intrinsics = [
    [[720.313, 0, 481.014], [0, 719.521, 360.991], [0, 0, 1]],
    [[768.113, 0, 472.596], [0, 767.935, 350.978], [0, 0, 1]],
    [[728.237, 0, 459.854], [0, 729.419, 351.590], [0, 0, 1]],
]

_distortions = [
    [0.395621, 0.633705, -2.41723, 2.11079],
    [0.368917, 1.50111, -7.94126, 11.9171],
    [0.276114, 2.09465, -9.97956, 14.1921],
]

# Construct cameraMat and distCoef
cameraMat = [np.array(K, dtype=np.float64) for K in _intrinsics]
distCoef = [np.array(D, dtype=np.float32).reshape(-1, 1) for D in _distortions]

if __name__ == "__main__":
    for i, (K, D) in enumerate(zip(cameraMat, distCoef)):
        print(f"Camera {i}:")
        print("  Intrinsic Matrix:\n", K)
        print("  Distortion Coefficients:\n", D)