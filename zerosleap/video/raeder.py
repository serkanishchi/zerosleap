import os
import cv2
import logging

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Provides more readable and controllable interface
    on top of OpenCV VideoCapture class.

    Args:
        filename: Name of the file
    """

    def __init__(self, filename):
        self._filename = filename

        # Check if the file is exists, if not raise error
        if not os.path.exists(self._filename):
            raise FileNotFoundError("Could not find the file:{}".format(self._filename))

        # Create a VideoCapture object to read frames from file
        self._reader = cv2.VideoCapture(self._filename)
        if not self._reader.isOpened():
            raise IOError("File can not opened.")

        # Read the first frame to get frame props
        ok, frame = self.read(0)
        if ok:
            self._width = frame.shape[1]
            self._height = frame.shape[0]
            self._channels = frame.shape[2]
        else:
            raise IOError(f'cannot read frame from {self._filename}.')

        # Reset to first frame
        self._seek(0)

    @property
    def frames_count(self):
        return self._reader.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def fps(self):
        return self._reader.get(cv2.CAP_PROP_FPS)

    @property
    def current_frame_pos(self):
        return self._reader.get(cv2.CAP_PROP_POS_FRAMES)

    @property
    def frame_width(self):
        return int(self._reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_height(self):
        return int(self._reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def fourcc(self):
        return int(self._reader.get(cv2.CAP_PROP_FOURCC))

    @property
    def frame_format(self):
        return int(self._reader.get(cv2.CAP_PROP_FORMAT))

    @property
    def frame_shape(self):
        return self._width, self._height, self._channels

    # Get a frame from video file with the index number
    def get_frame(self, idx: int) -> tuple:
        self._reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = self._reader.read()

        if not success:
            raise IOError("The frame {} cannot be loaded from {}".format(idx, self._filename))

        return success, frame

    # Reads the next frame from the video file
    def read(self, frame_number=None):
        """Read next frame or frame specified by `frame_number`."""

        # Check if we are at right position if seek the frame
        if frame_number is not None and not frame_number == self.current_frame_pos:
            self._seek(frame_number)

        # Read frame from reader and return it
        return self._reader.read()

    def _seek(self, frame_number):
        """Set the frame number to seek in video."""
        success = self._reader.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        if not success:
            raise IOError("The frame position cannot be changed to {}".format(frame_number))

    def _reset(self):
        """Reinit reader with filename."""
        self.__init__(self._filename)

    def close(self):
        self._reader.release()

    def __getitem__(self, index):
        """Enables slicing self[index] and self[start:stop:step]"""
        if isinstance(index, slice):
            return (self[ii] for ii in range(*index.indices(len(self))))
        elif isinstance(index, (list, tuple, range)):
            return (self[ii] for ii in index)
        else:
            return self.read(index)[1]

    def __len__(self):
        """Returns total number of frames"""
        return self.frames_count

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __repr__(self):
        return f"{self._filename} with {len(self)} frames of size {self.frame_shape} at {self.fps:1.2f} fps"
