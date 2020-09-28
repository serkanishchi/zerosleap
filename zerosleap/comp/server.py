"""
Module to compute functions in a parallel manner.

Provides computation from a separate process or a remote computation resource.

The files that the computation needed should be accessible from server side.
"""

import time
import logging
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from multiprocessing import Process

from typing import Optional, Union, Text, Dict

from zerosleap.conn.pair import PairServer
from zerosleap.processing.heatmap import find_heatmap_peaks

from zerosleap.processing.tracking.track import centroid_distance, Detection
from zerosleap.processing.tracking.tracker import Tracker

logger = logging.getLogger(__name__)


class ProcessingServer(Process):
    """
    ProcessingServer uses a bidirectional one to one "PAIR" messaging pattern
    as a server node to connect with client nodes that in need of computation.
    It provides a base server functionalities for parallel computing.
    ProcessingServers only can take or send serializable objects like
    numpy array or dictionary. Also can take only serializable parameters to
    be initialize.
        - Listens for requests.
        - Receives the incoming data
        - Make the computation
        - Send back computation outputs to the client.
    """

    def __init__(self, port: Optional[Union[Text, int]] = "9999"):
        super(ProcessingServer, self).__init__()
        self._metrics = {}
        self._port = port

    def build(self):
        """All the initialization process should be inside the build function"""

        # Communication node for receiving and sending data
        self._server = PairServer(f'tcp://*:{str(self._port)}')

        # Initialize server metrics.
        self._metrics = {}
        self.init_metrics()

        logger.info(f"Processing server initialized.")

    @abstractmethod
    def process(self, data: np.ndarray, args: Dict = None) -> Union[Dict, np.ndarray]:
        pass

    def run(self):
        # Initialization must be inside process's activity
        self.build()

        logger.info(f"Processing server started to running: {time.time()}")

        while True:
            # loop time start
            lts = time.time()
            self._metrics["iteration"] += 1

            try:
                ts = time.time()
                # Receive the data that should be computed.
                args, data = self._server.recv_array()
                te = time.time()

                # If receiving time bigger than 1 seconds reset metrics
                if te-ts > 1:
                    self.init_metrics()

                self._metrics["recv_time"] += te-ts
            except Exception:
                raise ConnectionError(
                    f"Can not received the data from client node."
                )

            # If stop request is came break the loop
            if "stop" in args:
                break

            ts = time.time()
            # Make the computation at the server node
            output = self.process(data, args)
            te = time.time()
            self._metrics["comp_time"] += te-ts

            try:
                ts = time.time()
                # Send the calculated output.
                if type(output) == np.ndarray:
                    pass
                elif type(output) == dict:
                    self._server.send_dict(output)
                else:
                    raise AssertionError(
                        f'Not supported output type {type(output)}.'
                    )
                te = time.time()
                self._metrics["send_time"] += te-ts
            except Exception:
                raise ConnectionError(
                    f"Can not send the data."
                )

            # loop time end
            lte = time.time()
            self._metrics["overall_time"] += lte-lts

    def init_metrics(self):
        """Initialize/Reset server metrics"""
        self._metrics["iteration"] = 1
        self._metrics["overall_time"] = 0.000001
        self._metrics["recv_time"] = 0
        self._metrics["send_time"] = 0
        self._metrics["comp_time"] = 0

    def close(self):
        """Close the connection."""
        logger.info(f"Processing server closed at {time.time()}")
        self._server.close()


class VideoProcessingServer(ProcessingServer):
    """
    Provides video processing as a separate process.
        - Loads the model for inference.
        - Listens for requests.
        - Receives the image or chunk of images.
        - Predict the heatmaps and calculate the peaks.
        - Sends back the results.
    """

    def __init__(self, _model_path, port="9999"):
        super(VideoProcessingServer, self).__init__(port)
        self._model_path = _model_path

    def build(self):
        """All the initialization process should be inside the build function"""
        super().build()
        # Load the model.
        self._model = tf.keras.models.load_model(self._model_path, compile=False)
        logger.info(f"Video Processing server initialized.")

    def process(self, data: np.ndarray, args: Dict = None, peaks=True):
        # Preprocess the image for inference.
        # Normalize and reshape the image data.
        X = np.expand_dims(data, axis=0).astype("float32") / 255.
        X = tf.image.resize(X[0], size=[512, 512])

        # Make the prediction.
        Y = self._model.predict(X)

        output = {}

        # Add heatmaps to the output if requested.
        # Should be returned as np.ndarray as in receiving image,
        # converting to list is slower approach.
        # Not recommended, its just for demonstration.
        if args["heatmaps"]:
            output["heatmaps"] = Y.squeeze().tolist()

        # Add peaks to the output if requested.
        if args["peaks"]:
            # Find local peaks.
            peaks = find_heatmap_peaks(Y)

            # Prepare the peaks for serialization.
            # No object is accepted only list and dict.
            peak_points, peak_vals, peak_sample_idx  = peaks
            peaks = {"points": peak_points.numpy().tolist(),
                     "peak_vals": peak_vals.numpy().tolist(),
                     "frame_idx": peak_sample_idx.numpy().tolist()}
            output["peaks"] = peaks

        # Add server metrics to the output if requested.
        if args["metrics"]:
            output["metrics"] = self._metrics

        return output


class TrackProcessingServer(ProcessingServer):
    """
    Provides tracking as a separate process.
        - Creates a tracker (just Kalman Filter currently).
        - Listens for requests.
        - Receives the detections data to make tracking.
        - Calculates the track information for each detection.
        - Sends back the results.
    """

    def __init__(self,
                 port: Optional[Union[Text, int]] = "5555",
                 input_type: Optional[Union[Dict, np.ndarray]] = np.ndarray):
        super(TrackProcessingServer, self).__init__(port)
        self.input_type = input_type

    def build(self):
        """All the initialization process should be inside the build function"""
        super().build()
        # Initialize tracker object with a cost function
        self._tracker = Tracker(distance_function=centroid_distance, distance_threshold=20)
        logger.info(f"Track Processing server initialized.")

    def process(self, data: np.ndarray, args: Dict = None):
        # Reinitialize the tracker if "reset" request received
        if args["reset"] == True:
            self._tracker = Tracker(distance_function=centroid_distance, distance_threshold=20)

        # Generate Detection objects from input data [x, y]
        detections = []
        for point in data:
            detections.append(Detection(point))

        # Update the tracker with detections and assign each detection to tracked objects
        tracked_objects = self._tracker.update(detections)

        # Prepare the tracked objects for serialization
        points = {}
        for obj in tracked_objects:
            points[obj.id] = obj.estimate.tolist()

        output = {"tracks": points}

        # Add tracks to the output.

        # Add metrics to the output if requested.
        if args["metrics"]:
            output["metrics"] = self._metrics

        return output
