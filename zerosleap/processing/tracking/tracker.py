import numpy as np

from zerosleap.processing.tracking.track import Track, Detection


class Tracker:
    """
    Object used for tracking of detections between sequential frames. Initiated and updated with
    detected peak points.

    Attributes:
        tracked_objects: Keeps tracked detections.
        min_drop_life: Defines the minimum life of the undetected tracks before drop
        max_drop_life: Defines the maximum life of any undetected tracks before drop
        distance_threshold: Maximum distance for new detections to be considered as tracked object.
        confidence_threshold: Used to mask the detections that have confidence lower than the threshold value
    """
    def __init__(
        self,
        distance_function,
        distance_threshold,
        min_drop_life=10,
        max_drop_life=25,
        confidence_threshold=0,
    ):

        """
        Initialize the Tracker

        Args:
            distance_function: Function to calculate distance between tracks and detections
            distance_threshold: Threshold for new detections to be considered as tracked object.
            min_drop_life: Remaining life of the undetected tracks before drop
            max_drop_life: Max remaining life of any undetected tracks before drop
            confidence_threshold: Threshold for detections to be considered as real detection
        """

        self.tracked_objects = []
        self.distance_function = distance_function
        self.min_drop_life = min_drop_life
        self.max_drop_life = max_drop_life
        self._distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        Track.count = 0

    def update(self, detections: Detection = None) -> list:
        """
        Update tracks according to the detections.

        Args:
            detections: Detected objects

        Returns:
            List of already initialized tracked objects
        """

        # Get the stable trackers that have momentum
        self.tracked_objects = [o for o in self.tracked_objects if o.has_momentum]

        # Update tracker
        for obj in self.tracked_objects:
            obj.tracker_step()

        # Update already initialized tracked objects with detections
        unmatched_detections = self.update_tracks(
            [o for o in self.tracked_objects if not o.initializing], detections
        )

        # Update not yet initialized tracked objects with yet unmatched detections
        unmatched_detections = self.update_tracks(
            [o for o in self.tracked_objects if o.initializing], unmatched_detections
        )

        # Create new tracked objects from remaining unmatched detections
        for detection in unmatched_detections:
            self.tracked_objects.append(
                Track(
                    detection,
                    self.min_drop_life,
                    self.max_drop_life,
                    self.confidence_threshold,
                )
            )

        return [p for p in self.tracked_objects if not p.initializing]

    def update_tracks(self, tracks: [Track], detections: [Detection]) -> [Track]:
        """
        Update tracks in place and returns unmatched detections

        Args:
            tracks: List of tracks to be matched
            detections: List of detections to be matched
        """

        if detections is not None and len(detections) > 0:

            # Creates a distance matrix of ones for each detection and track
            distance_matrix = np.ones((len(detections), len(tracks)), dtype=np.float32)

            # Scale the distance matrix with _distance_threshold
            distance_matrix *= self._distance_threshold + 1

            for d, detection in enumerate(detections):
                for t, track in enumerate(tracks):
                    # Calculate the distance between each detection and track
                    distance = self.distance_function(detection, track)

                    if distance > self._distance_threshold:
                        distance_matrix[d, t] = self._distance_threshold + 1
                    else:
                        distance_matrix[d, t] = distance

            if np.isnan(distance_matrix).any():
                raise ValueError(
                    f"Distance function returned nan value."
                )
            if np.isinf(distance_matrix).any():
                raise ValueError(
                    f"Distance function returned inf value."
                )

            # Update tracks current_min_distance for checking
            if distance_matrix.any():
                for i, minimum in enumerate(distance_matrix.min(axis=0)):
                    tracks[i].current_min_distance = (
                        minimum if minimum < self._distance_threshold else None
                    )

            # Get indices of the matched tracks and detections
            matched_det_indices, matched_track_indices = self.match(
                distance_matrix
            )

            if len(matched_det_indices) > 0:
                # Create unmatched_detections
                unmatched_detections = [
                    d for i, d in enumerate(detections) if i not in matched_det_indices
                ]

                # Handle matched tracks/detections
                for (match_det_idx, match_track_idx) in zip(
                    matched_det_indices, matched_track_indices
                ):
                    match_distance = distance_matrix[match_det_idx, match_track_idx]
                    matched_detection = detections[match_det_idx]
                    matched_tracks = tracks[match_track_idx]

                    if match_distance < self._distance_threshold:
                        # If match distance lower than the _distance_threshold
                        # Update last distance
                        matched_tracks.add(matched_detection)
                        matched_tracks.last_distance = match_distance
                    else:
                        # If distance higher than the threshold add the detection
                        # to unmatched_detections
                        unmatched_detections.append(matched_detection)
            else:
                # If there is no detected matches,
                # add all detections to unmatched_detections.
                unmatched_detections = detections
        else:
            # If there is no detection unmatched_detections
            # also will be empty list
            unmatched_detections = []

        # Return unmatched detections
        return unmatched_detections

    def match(self, distance_matrix: np.ndarray) -> [list, list]:
        """
        Matches detections with tracked_objects from a distance matrix

        Args:
             distance_matrix: Distance matrix between tracked_objects and detections

        Returns:
             det_ids: List of matched detection indexes
             track_ids: List of matched track indexes
        """
        distance_matrix = distance_matrix.copy()
        if distance_matrix.size > 0:
            det_ids = []
            track_ids = []
            current_min = distance_matrix.min()

            # This part should be vectorized
            while current_min < self._distance_threshold:
                # Get the index of minimum values
                flattened_arg_min = distance_matrix.argmin()

                # Find index of detections
                det_idx = flattened_arg_min // distance_matrix.shape[1]

                # Find index of tracks
                track_idx = flattened_arg_min % distance_matrix.shape[1]

                det_ids.append(det_idx)
                track_ids.append(track_idx)
                distance_matrix[det_idx, :] = self._distance_threshold + 1
                distance_matrix[:, track_idx] = self._distance_threshold + 1
                current_min = distance_matrix.min()

            return det_ids, track_ids
        else:
            return [], []
