import numpy as np
import math
from sklearn import linear_model
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


# Get the distance between points to lines
def getDistance2Line(lines, pts):
    """
    Compute the perpendicular distance from points to lines.

    Parameters
    ----------
    lines : array_like, shape (N, 3)
        Each line represented as (a, b, c) for ax + by + c = 0.
    pts : array_like, shape (M, 2)
        2D points to compute distance from.

    Returns
    -------
    inliers : ndarray, shape (M,)
        Boolean array indicating which points are within threshold (hardcoded 5).
    distances : ndarray, shape (M,)
        Computed perpendicular distances for each point.
    """
    pts, out, lines = np.copy(pts).reshape(-1, 2), [], np.copy(lines).reshape(-1, 3)

    for [a, b, c] in lines:
        for [x, y] in pts:
            out.append(abs(a * x + b * y + c) / np.sqrt(np.sqrt(pow(a, 2) + pow(b, 2))))

    return np.array(out) < 5, np.array(out)


# Find a plane that passes between three points
def findPlane(P1, P2, P3):
    """
    Fit a plane through three 3D points.

    Parameters
    ----------
    P1, P2, P3 : array_like, shape (3,)
        3D coordinates of the points.

    Returns
    -------
    plane : ndarray, shape (4,)
        Plane coefficients (a, b, c, d) for ax + by + cz + d = 0.
    """
    x1, y1, z1 = P1
    x2, y2, z2 = P2
    x3, y3, z3 = P3
    a1, b1, c1 = x2 - x1, y2 - y1, z2 - z1
    a2, b2, c2 = x3 - x1, y3 - y1, z3 - z1
    a, b, c = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - b1 * a2
    d = -a * x1 - b * y1 - c * z1
    return np.array([a, b, c, d])


# Get angle between two vectors
def getAngle(u, v):
    """
    Compute angle between two vectors.

    Parameters
    ----------
    u, v : array_like
        Input vectors.

    Returns
    -------
    angle : float
        Angle between vectors in radians, always in [-pi, pi].
    """
    cosPhi = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    phi = np.arccos(cosPhi)
    return np.arctan2(np.sin(phi), cosPhi)


def reshapeCoord(coord):
    """
    Reshape 1D or 2D coordinate array into separate X and Y arrays.

    Parameters
    ----------
    coord : array_like, shape (N*2,) or (N, 2)
        Input coordinates.

    Returns
    -------
    coordX : ndarray
        X-coordinates.
    coordY : ndarray
        Y-coordinates.
    """
    return np.asarray(coord).reshape(-1, 2).T


def normalizePoints(pts):
    """
    Compute angle between two vectors.

    Parameters
    ----------
    u, v : array_like
        Input vectors.

    Returns
    -------
    angle : float
        Angle between vectors in radians, always in [-pi, pi].
    """
    # Calculate origin centroid
    center = np.mean(pts, axis=0)

    # Translate points to centroid
    traslatedPts = pts - center

    # Calculate scale for the average point to be (1,1,1) >> homogeneous
    meanDist2Center = np.mean(np.linalg.norm(traslatedPts, axis=1))
    if meanDist2Center:  # Protect against division by zero
        scale = np.sqrt(2) / meanDist2Center
    else:
        return pts, 0, False

    # Compute translation matrix >> (x-x_c)*scale
    T = np.diag((scale, scale, 1))
    T[0:2, 2] = -scale * center

    # Transform in homogeneous coordinates
    homogeneousPts = np.vstack((pts.T, np.ones((1, pts.shape[0]))))
    normPoints = np.matmul(T, homogeneousPts)

    return normPoints, T, True


def singularValueDecomposition(matrix):
    """
    Compute SVD and return diagonalized singular value matrix.

    Parameters
    ----------
    matrix : ndarray, shape (M, N)
        Matrix to decompose.

    Returns
    -------
    U : ndarray, shape (M, M)
    D : ndarray, shape (M, N)
    V : ndarray, shape (N, N)
    """
    leftSingVectors, singValues, rightSingVectorsTransposed = np.linalg.svd(matrix)
    singValuesMatrix = np.zeros((3, 3))

    np.fill_diagonal(singValuesMatrix, singValues)
    rightSingVectors = rightSingVectorsTransposed.T.conj()

    return leftSingVectors, singValuesMatrix, rightSingVectors


def isCollinear(P1, P2, P3, max_ratio=0.08, min_distance=10.0):
    """
    Determine whether three 2D points are nearly collinear.

    Parameters
    ----------
    P1, P2, P3 : array_like, shape (2,)
        The three 2D points.
    max_ratio : float, optional
        Max allowed perpendicular distance ratio.
    min_distance : float, optional
        Minimum distance between any two points.

    Returns
    -------
    collinear : bool
        True if points are collinear.
    """
    P1, P2, P3 = np.array(P1), np.array(P2), np.array(P3)

    max_point_dist = np.max(
        [np.linalg.norm(P2 - P1), np.linalg.norm(P3 - P1), np.linalg.norm(P3 - P2)]
    )
    min_point_dist = np.min(
        [np.linalg.norm(P2 - P1), np.linalg.norm(P3 - P1), np.linalg.norm(P3 - P2)]
    )
    # Reject degenerate cases (points too close to each other)
    if min_point_dist < min_distance:
        return False

    # Vector from P1 to P2
    line_vec = P2 - P1
    line_unit = line_vec / np.linalg.norm(line_vec)

    # Vector from P1 to P3
    vec_to_P3 = P3 - P1

    # Project and compute perpendicular distance
    proj_length = np.dot(vec_to_P3, line_unit)
    proj_point = P1 + proj_length * line_unit
    perp_dist = np.linalg.norm(P3 - proj_point)

    # Compute relative ratio
    ratio = perp_dist / max_point_dist
    return ratio < max_ratio


# Interpolate data using cubic spline
def interpolate(coords, timestamps, steps):
    """
    Perform cubic spline interpolation of coordinate data over timestamps.

    Parameters
    ----------
    coords : array_like, shape (N, ...)
        Coordinate values to interpolate.
    timestamps : array_like, shape (N,)
        Corresponding timestamps.
    steps : int
        Time step interval.

    Returns
    -------
    interpolated_coords : ndarray
        Interpolated coordinates.
    newTimestamps : ndarray
        New timestamps after interpolation.
    """
    # Get data
    if not len(timestamps):
        return [], []

    # Get array limits
    lowBound = math.ceil(timestamps[0] / steps)
    highBound = math.floor(timestamps[-1] / steps)

    # Interpolate
    newTimestamps = np.linspace(
        lowBound, highBound, int((highBound - lowBound)) + 1, dtype=np.uint16
    )
    cubicSpline = CubicSpline(timestamps, coords, axis=0)

    return cubicSpline(newTimestamps * steps), newTimestamps


def getSignal(n1, n2, tol=1e-6):
    """
    Determine the sign of difference between two numbers with tolerance.

    Parameters
    ----------
    n1, n2 : float
        The numbers to compare.
    tol : float, optional
        Tolerance for treating them as equal.

    Returns
    -------
    signal : int
        -1 if n1 < n2, +1 if n1 > n2, 0 if equal within tolerance.
    valid : bool
        False if numbers are considered equal, True otherwise.
    """
    if abs(n1 - n2) <= tol:
        return 0, False
    return (-1 if (n1 - n2) < 0 else 1), True


def swapElements(arr, idx1, idx2):
    """
    Swap two elements in a list or array.

    Parameters
    ----------
    arr : list or ndarray
        Array to modify.
    idx1, idx2 : int
        Indices to swap.

    Returns
    -------
    arr : list or ndarray
        Modified array after swapping.
    """
    arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
    return arr
