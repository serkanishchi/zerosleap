"""
Objects and functions will be used by Tracker
"""

import math
import numpy as np
from filterpy.kalman import KalmanFilter


class Detection:
    """
    Detected key points (peaks).

    Attributes:
        points: Numpy array [n x 2] x, y for each point.
        confidence: Confidence of each peak points [n x score], calculated from heatmap.
    """

    def __init__(self, points, confidence=None):
        self.points = points
        self.confidence = confidence


class Track:
    """
    Tracks object detected by the tracker. Track object uses Kalman Filter
    to keep and estimate track parameters.
    """

    count = 0

    def __init__(
            self,
            detections,
            min_drop_life,
            max_drop_life,
            confidence_threshold,
    ):
        # Age of the track
        self.age = 0

        # Used for filtering points that have confidence lower than the threshold value
        self.confidence_threshold = confidence_threshold

        # Number of points detected
        self.points_count = validate_points_shape(detections.points).shape[0]

        # Minimum life of the undetected track before drop
        self.min_drop_life = min_drop_life

        # Maximum life of the undetected track before drop
        self.max_drop_life = max_drop_life

        # Minimum life of a point before drop, more volatile than track
        self.point_min_drop_life = math.floor(min_drop_life / 4)

        # Maximum life of a point before drop
        self.point_max_drop_life = math.ceil(max_drop_life / 4)

        # If min and max drop life of a point equals, increase the max
        if (self.point_max_drop_life - self.point_min_drop_life) < 1:
            self.point_max_drop_life = self.point_min_drop_life + 1

        # Init detection_count with min_drop_life + 1
        # to prevent dropping new tracks
        self.detection_count = min_drop_life + 1

        # Init detection_count for each point
        self.point_detection_count = np.ones(self.points_count) * self.point_min_drop_life

        # Last distance to detections
        self.last_distance = None

        # Current min distance to detections
        self.current_min_distance = None

        # Last detections
        self.last_detection = detections

        # Flag for not initialized tracks
        self.initializing_flag = True

        # Id of the track
        self.id = None

        # Setup Kalman Filter
        self.setup_filter(detections.points)

        # Init detected-at_least_once_points with False
        self.detected_at_least_once_points = np.array([False] * self.points_count)

    def setup_filter(self, detections: np.ndarray):
        """
        Initialize Kalman Filter

        Args:
            detections: Points for initializing Kalman Filter
        """

        # Validate shape of detected points
        detections = validate_points_shape(detections)

        # Calculates dimension of 'x' (Filter State Estimate)
        # Position x Velocity x Number of Points
        dim_x = 2 * 2 * self.points_count

        # Calculates dimension of 'z' (Last Measurement)
        dim_z = 2 * self.points_count
        self.dim_z = dim_z

        # Initialize the Kalman Filter
        self.filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

        # Define the F (State Transition Matrix)
        self.filter.F = np.eye(dim_x)

        # Update
        dt = 1  # At each step we update pos with v * dt
        for p in range(dim_z):
            self.filter.F[p, p + dim_z] = dt

        # Define the H (Measurement Function)
        self.filter.H = np.eye(
            dim_z,
            dim_x,
        )

        # Define the R (Measurement Uncertainty)
        self.filter.R *= 4.0

        # Define the Q (Process Uncertainty/Noise)
        self.filter.Q[dim_z:, dim_z:] /= 10

        # Initial state: numpy.array(dim_x, 1)
        self.filter.x[:dim_z] = np.expand_dims(detections.flatten(), 0).T

        # Estimation uncertainty: numpy.array(dim_x, dim_x)
        self.filter.P[dim_z:, dim_z:] *= 10.0

    def tracker_step(self):
        """Tracker step function to make filter predict"""
        self.detection_count -= 1
        self.point_detection_count -= 1
        self.age += 1
        # Advances the tracker's state
        self.filter.predict()

    @property
    def initializing(self):
        if (
                self.initializing_flag
                and self.detection_count > (self.min_drop_life + self.max_drop_life) / 2
        ):
            self.initializing_flag = False
            Track.count += 1
            self.id = Track.count
        return self.initializing_flag

    @property
    def has_momentum(self) -> bool:
        """
        If track detection_count is higher than the
        min_drop_life it means the track is live

        Returns:
            If track has momentum or alive
        """
        return self.detection_count >= self.min_drop_life

    @property
    def estimate(self) -> np.ndarray:
        """
        Gets estimate positions of detections from x (Filter State estimate)

        Returns:
            positions: Estimate position of the detections
        """

        positions = self.filter.x.T.flatten()[: self.dim_z].reshape(-1, 2)
        return positions

    def add(self, detection: Detection):
        """
        Adds the matched detection to the track and updates the filter.
        Also updates all counts.

        Args:
            detection: Matched detection to be added the track
        """
        points = validate_points_shape(detection.points)

        self.last_detection = detection
        if self.detection_count < self.max_drop_life:
            self.detection_count += 2

        # If detection confidence is used to model track state
        if detection.confidence is not None:

            # Check the shape of the detection confidence
            assert len(detection.confidence.shape) == 1

            # Creates a mask for points confidence over confidence_threshold value
            points_over_threshold_mask = detection.confidence > self.confidence_threshold

            # Generate the threshold mask
            threshold_mask = np.array(
                [[m, m] for m in points_over_threshold_mask]
            ).flatten()

            # Measurement function positions
            H_pos = np.diag(threshold_mask).astype(float)

            # Increase the detected points count
            self.point_detection_count[points_over_threshold_mask] += 2
        else:
            # Detection confidence is not used just create a mask with True values
            points_over_threshold_mask = np.array([True] * self.points_count)

            # Measurement function positions
            H_pos = np.identity(points.size)

            # Increase the detected points count
            self.point_detection_count += 2

        # Limit the count of point detection to point max drop life
        # This will be used to define how long this object can live
        # without getting matched to any detections.
        self.point_detection_count[
            self.point_detection_count >= self.point_max_drop_life
            ] = self.point_max_drop_life

        # Assign 0 to all points detection counts lower than 0
        self.point_detection_count[self.point_detection_count < 0] = 0

        # Measurement function velocity
        H_vel = np.zeros(H_pos.shape)

        # Creates measurement function by stacking position and velocity parts
        H = np.hstack([H_pos, H_vel])

        # Adds a new measurement to the filter with provided measurement function
        self.filter.update(np.expand_dims(points.flatten(), 0).T, None, H)

        # Creates a mask for detected points at least once
        detected_at_least_once_mask = np.array(
            [[m, m] for m in self.detected_at_least_once_points]
        ).flatten()

        # To prevent huge velocity vectors when first real detection occurs, force the
        # velocity = 0 for the first time.
        self.filter.x[self.dim_z:][np.logical_not(detected_at_least_once_mask)] = 0

        # Update detected_at_least_once_points
        self.detected_at_least_once_points = np.logical_or(
            self.detected_at_least_once_points, points_over_threshold_mask
        )


def validate_points_shape(points: np.ndarray) -> np.ndarray:
    """
    Validates points shapes

    Args:
        points: A numpy array with the shape of (2, )

    Returns:
        points:
    """

    # If tracked only one point
    if points.shape == (2,):
        # Reshape it
        points = points[np.newaxis, ...]
    else:
        # Check the shape of the points
        if points.shape[1] != 2 or len(points.shape) > 2:
            raise AssertionError(
                f"Not matched shape for detection points. Detections shape should be [n x 2] not {points.shape}"
            )

    return points


# Distance function
def centroid_distance(detection, tracked_object) -> np.ndarray:
    return np.linalg.norm(detection.points - tracked_object.estimate)
