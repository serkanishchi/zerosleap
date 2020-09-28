"""
Manages video reading, video processing, track
processing and compose raw frames with processed
data. Acts as a provider for video player.

This class is a content producer for video player.
And generates frames with raw and processed data.
"""

from threading import Thread
import time
from queue import Queue
import numpy as np

from zerosleap.comp.processor import VideoProcessor
from zerosleap.gui.metaframe import MetaFrame
from zerosleap.video.raeder import VideoReader


class VideoComposer:

    def __init__(self, path, buffer_size=256, chunk_size=32):
        """"
        Initialize video composer.

        Args:
            path: Video filename path.
            buffer_size:
            chunk_size:
        """
        self.video_reader = VideoReader(path)

        # Asynchronously processing raw video frames with chunks.
        # If the process is not complete, not blocks update loop.
        # This is a consumer of video_reader object.
        # Chunk is necessary for improving processing speed
        # especially at GPU.
        self.video_processor = VideoProcessor(9999)

        # Buffer for raw video frames
        self._buffer = []

        # Buffer for the processed frames
        # Keeps also peaks, tracks and heatmaps (optional)
        self._meta_frames = Queue(maxsize=buffer_size)

        self._run_flag = True
        self._reset_buffer_flag = False
        self._frame_index_changed =False

        # Request for heatmap
        self._heatmaps_flag = False

        self._chunk_size = chunk_size
        self._frame_index = 0

        # intialize the thread with the update function
        # Update function is a non blocking control loop
        # Except file reading
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        """Generate extended frames with raw and processed data."""
        # Controller loop
        while self._run_flag:

            # If frame index changed manually and _reset_buffer_flag is set
            # empty the _frames queue and _buffer
            if self._reset_buffer_flag:
                with self._meta_frames.mutex: self._meta_frames.queue.clear()
                self._buffer = []
                self._reset_buffer_flag = False

            # Continue to grab images until the _frames queue is full
            if not self._meta_frames.full():

                # Prevent unnecessary index changing
                frame_index = None
                if self._frame_index_changed:
                    frame_index = self._frame_index
                    self._frame_index_changed = False

                # read the next frame from the file
                (grabbed, frame) = self.video_reader.read(frame_index)

                # If the reader reaches end of the file and
                # the _frames queue is empty wait for another action
                if not grabbed and self._buffer == []:
                    time.sleep(0.1)
                    continue
                else:
                    self._frame_index += 1

                # If frame is grabed from video reader
                # Add frames to the buffer for processing with chunk
                if grabbed:
                    self._buffer.append(frame[:, :, :1])

                result = None

                # If size of the buffer bigger than the chunk size
                # or if we reached end of the file and the size of the
                # buffer bigger than 0, process the frames
                if len(self._buffer) >= self._chunk_size or \
                        (not grabbed and len(self._buffer) >= 0):

                    # Try to get processed frames from processing server
                    result = self.video_processor.recv()

                    # Keeps raw frames as global for adding to _frames
                    frames = []

                    # If the results ready from the processing server
                    if result is not None:
                        result_length = len(result["peaks"])
                        # Take the processed raw frames
                        frames = self._buffer[:result_length]
                        # Remove them from the buffer.
                        self._buffer = self._buffer[result_length:]

                    # Try to get a chunk from buffer
                    chunk = None
                    # If buffer size bigger than the chunk size
                    # Take the chunk and send it to the video processor
                    if len(self._buffer) >= self._chunk_size:
                        self.video_processor.send(np.stack(self._buffer[:self._chunk_size], axis=0),
                                                  peaks=True,
                                                  heatmaps=self._heatmaps_flag)
                    # If the buffer size lower than the chunk size and
                    # we reached end of the file, take the rest
                    elif len(self._buffer) > 0 and not grabbed:
                        self.video_processor.send(np.stack(self._buffer[:], axis=0),
                                                  peaks=True,
                                                  heatmaps=self._heatmaps_flag)
                    # If there is no frame in the _buffer just continue to
                    # wait in the loop until somebody changed the _run_flag
                    # or changed the _frame_index.
                    else:
                        time.sleep(0.1)
                        continue

                    if result is not None:
                        # take peaks from the result
                        peaks = result["peaks"]
                        heatmaps = None
                        if "heatmaps" in result:
                            heatmaps = result["heatmaps"]

                        # Create Frame object for each result and add to _frames queue
                        for i in range(len(frames)):
                            if heatmaps is not None:
                                frame = MetaFrame(frame=frames[i],
                                                peaks=peaks[i],
                                                heatmap=heatmaps[i])
                            else:
                                frame = MetaFrame(frame=frames[i],
                                                  peaks=peaks[i])
                            self._meta_frames.put(frame)

    def read(self) -> MetaFrame:
        """Reads next frame in _frames queue"""
        return self._meta_frames.get()

    # Change the frame_index and reset all buffers with setting _reset_buffer flag
    def seek(self, frame_index: int):
        """
        Changes active _frame_index

        Args:
            frame_index: target frame index number for seeking
        """
        # Set _frame_index
        self._frame_index = frame_index
        # Set _frame_index_changed flag
        self._frame_index_changed = True
        # Set _rest_buffer_flag
        self._reset_buffer_flag = True

    def toggle_heatmap(self):
        """Toggle _heatmaps_flag"""
        self._heatmaps_flag = not self._heatmaps_flag

    def stop(self):
        # Set _run_flag as False to exit thread loop
        self._run_flag = False
        time.sleep(1)
        self.video_processor.stop()
