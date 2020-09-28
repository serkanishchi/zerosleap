"""
Module for controlling video streaming and handling the
user interface requests. Manages the video composer and
tracker objects. Provides stream and interactions between
processor objects and video player.
"""

from PySide2.QtCore import QThread, Signal
import time

from zerosleap.comp.processor import TrackProcessor
from zerosleap.gui.composer import VideoComposer
from zerosleap.gui.metaframe import MetaFrame


class VideoController(QThread):
    """
    Background thread for streaming processed video and
    controlling user interactions.

    Signals:
        scene_changed: Emitter for updating scene plot
        tracks_updated: Emitter for updating track table
        player_paused: Emitter for updating player status as paused
        seeker_value_updated: Emitter for updating seeker value
        seeker_range_updated: Emitter for updating seeker range
        server_stats_updated: Emitter for updating server stats
    """

    scene_changed = Signal(MetaFrame)
    tracks_updated = Signal(dict)
    player_paused = Signal(bool)
    seeker_value_updated = Signal(int)
    seeker_range_updated = Signal(int)
    server_stats_updated = Signal(str, str)

    def __init__(self, file_name: str):
        """
        Initialize VideoController object

        args:
            file_name: Name of the video file for providing video composer
        """
        super().__init__()

        # Track processor for processing peak data for each iteration
        self._track_processor = TrackProcessor(9998)

        self._run_flag = True

        # Start the SceneComposer with a given file name and chunk_size.
        # If GPU available chunk size can be increased for more smooth
        # user experience. Also can be used with CPU but should be adjusted
        # carefully. Automatically adjusting chunk size according to processing speed!
        self._composer = VideoComposer(file_name, chunk_size=1).start()

        # Keeps the current status of the player paused or not
        self._paused = False

        # Keeps a _frame_index number for video controller
        self._frame_index = 0

    def run(self):
        # Initialize and emit the seeker range to show correctly
        self.seeker_range_updated.emit(int(self._composer.video_reader.frames_count))

        # If the stream changed with user interaction
        # this flag will be set to reinit processors
        self._frame_index_changed = False

        # Thread loop, continues to work until _run_flag is False
        while self._run_flag == True:

            if self._paused:
                # If video player paused, but _frame_index not changed
                # Just sleep and continue to next loop
                if not self._frame_index_changed:
                    time.sleep(0.1)
                    continue
                # If vide player paused and _frame_index_changed
                # Keep working for this loop and show the current frame
                else:
                    self._frame_index_changed = False

            # Reads the next meta frame object
            meta_frame = self._composer.read()

            if meta_frame is not None:
                # Process tracks according to the scene (peaks)
                self._track_processor.send(meta_frame.peaks)

                # Update the tracks in the scene
                meta_frame.tracks = self._track_processor.recv["tracks"]

                if self._frame_index % 24 == 0:
                    # Update tracks table at each 24 frames (second)
                    self.tracks_updated.emit(meta_frame.tracks)

                    # Update server metrics
                    self.server_stats_updated.emit(self._composer.video_processor.server_summary(),
                                                   self._track_processor.server_summary())

                # Emit the scene data for redrawing
                self.scene_changed.emit(meta_frame)

                # If not reached to the end of file (eof), increase the _frame_index
                # and update the seeker
                if self._frame_index < self._composer.video_reader.frames_count:
                    self._frame_index += 1
                    self.seeker_value_updated.emit(self._frame_index)

            # If reached end of the file (eof)
            if self._frame_index >= self._composer.video_reader.frames_count-1:
                # Change player status as _pause
                self._paused = True
                self.player_paused.emit(True)

    def set_frame(self, frame_index: int):
        """
        Changes frame index manually. Activates with
        changing value of seeker.

        Args:
            frame_index: target frame index number for seeking
        """

        # Just reset the track_processor and all track models.
        self._track_processor.reset_tracks()

        # Change _composer frame index number
        self._composer.seek(frame_index)

        # Set self._frame_index
        self._frame_index = frame_index

        # Set _frame_index_changed flag
        self._frame_index_changed = True

    def stop(self):
        """Make the thread exit from loop and close safely."""
        # Set _run_flag as False
        self._run_flag = False
        time.sleep(0.1)

        # Stop the composer
        self._composer.stop()

        # Stop the tracker
        self._track_processor.stop()
