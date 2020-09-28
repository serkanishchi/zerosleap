import numpy as np


class MetaFrame:
    """
    MetaFrame object is keeps just one moment of the produced scene.
    It differs from raw video frame. Keeps raw video data with processed
    additional data.

    Attributes:
        frame: Raw video frame.
        peaks: Peaks extracted from heatmap.
        heatmap: Processed frame with inference model.
        tracks: Track traces over time.
    """

    frame: np.ndarray
    heatmap: np.ndarray
    peaks: dict
    tracks: dict

    def __init__(self, frame, peaks, tracks=None, heatmap=None):
        self.frame = frame
        self.heatmap = heatmap
        self.peaks = peaks
        self.tracks = tracks

    def compare(self, ground_truth):
        """Will be used compare with a ground truth track data."""
        pass
