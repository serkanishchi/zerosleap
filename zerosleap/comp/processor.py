"""
Computation Module for parallel processing.

This module using ZeroMQ for messaging between processes. It encapsulates
both (server and client) part of the parallel processing and provides a
simple interface for usage.

Can be used extending "Processor" Abstract Base Class with overriding
    - send(self, data, args)
    - receive(self)
    - stop(self)
base functions.

Processor class needs a ProcessingServer class as an attribute which
will make the actual processing.
"""
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np
import os.path

from zerosleap.comp.server import VideoProcessingServer, TrackProcessingServer, ProcessingServer
from zerosleap.conn.pair import PairClient


class Processor(ABC):
    """
    Abstract Base Class for parallel processing module that cover all the
    communication and computation process at client application side.
    """

    _client: PairClient = None
    _server: ProcessingServer = None

    def __init__(self):
        super().__init__()

    @abstractmethod
    def send(self, data, args):
        pass

    @abstractmethod
    def recv(self):
        pass

    @abstractmethod
    def stop(self):
        pass


class VideoProcessor(Processor):
    """
    Take care of all parallel video processing process.
        - Send images to the video processing server.
        - Receives the results from video processing server.
        - Manages the communication between client and server.
    """

    def __init__(self, port):
        super(VideoProcessor).__init__()

        # Get package main folder path
        current_path = os.path.dirname(sys.modules['__main__'].__file__)
        # The model should be inside of the resources folder
        model_path = os.path.join(current_path, "../resources/keras_model.h5")

        # Initialize VideoProcessingServer with model. The model path
        # should be accessible from server side. Don't have any access
        # to server object from client side just managing its lifecycle.
        self._server = VideoProcessingServer(model_path, port=port)
        self._server.start()
        self._server_metrics = {}

        # Initialize the client with a PairClient object
        self._client = PairClient(address=f"tcp://127.0.0.1:{port}")

        # Used to check processor status, send/recv mode.
        self._send_flag = False

    def send(self, data: np.ndarray, peaks=True, heatmaps=False):
        """
        Sends image or chunk of images to the server with
        additional parameters.

        Args:
            data: n x image as numpy array [n, width, height, channel]
            peaks: request for peak
            heatmaps: request for heatmaps
        """

        args = {"peaks": peaks,
                "heatmaps": heatmaps,
                "metrics": True}

        self._client.send_array(args, data)

        # Set _send_flag to True for synchronization
        self._send_flag = True

    def recv(self) -> Union[Dict, None]:
        """
        Receive processed images data.

        Returns:
            None if _send_flag is not set, otherwise
            Dict:
                "peaks" : []
                "heatmaps" : []
                "metrics" : []
        """
        # If _send_flag is not set return None
        if not self._send_flag:
            return None

        result = self._client.recv_dict()
        output = {}

        if "peaks" in result:
            peaks = result["peaks"]

            # Convert peak points to points list for each frame.
            # Cause of only serialized data types can be used for
            # communication and to separating data processing and
            # data manipulation functions data manipulations should be
            # in client side.
            peak_points = [None] * len(set(peaks["frame_idx"]))
            for i, idx in enumerate(peaks["frame_idx"]):
                if peak_points[idx] is None:
                    peak_points[idx] = []
                peak_points[idx].append(peaks["points"][i])
            output["peaks"] = peak_points

        if "heatmaps" in result:
            output["heatmaps"] = np.asarray(result["heatmaps"])

        # Update server metrics
        self._server_metrics = result["metrics"]

        # Set _send_flag as False
        self._send_flag = False

        return output

    def server_metrics(self) -> dict:
        """
        Returns server metrics.

        Returns:
            output: Server metrics as dictionary
                ["recv"]: Receive time mean
                ["comp"]: Computation time mean
                ["send"]: Sending time mean
                ["load"]: Computation Time / Overall Time
        """
        output = {"recv": self._server_metrics["recv_time"] / self._server_metrics["iteration"],
                  "comp": self._server_metrics["comp_time"] / self._server_metrics["iteration"],
                  "send": self._server_metrics["send_time"] / self._server_metrics["iteration"],
                  "load": self._server_metrics["comp_time"] / self._server_metrics["overall_time"]}

        return output

    def server_summary(self) -> str:
        metrics = self.server_metrics()
        summary = ""
        summary += f"Video Processing Server | Load: {metrics['load']:.3f} | Comp: {metrics['comp']:.3f}   "
        return summary

    def stop(self):
        """Close and terminate all used sockets"""
        # If there is any waiting recv, just receive it
        if self._send_flag:
            self.recv()

        # Send the server stop request
        args = {"stop": True}
        self._client.send_array(args, np.ndarray([1, 1, 1]))

        # Close the connection
        self._client.close()

        # Terminate the server process
        self._server.terminate()


class TrackProcessor(Processor):
    """
    Take care of parallel tracking process.
        - Send detections to the track processing server.
        - Receives the results from track processing server.
        - Manages the communication between client and server.
    """

    def __init__(self, port):
        super(Processor).__init__()
        # Initialize the TrackProcessingServer, don't have any
        # access to server object from client side. Just managing
        # its lifecycle.
        self._server = TrackProcessingServer(port=port)
        self._server.start()
        self._server_metrics = {}

        # Initialize the client with a PairClient object
        self._client = PairClient(address=f"tcp://127.0.0.1:{port}")

        # Keep tracks information
        self.tracks = {}

        # Used to check processor status, send/recv mode.
        self._send_flag = False

        # Used to reset tracking processing on the server side
        self._reset_flag = False

    def send(self, data: np.ndarray):
        """
        Sends detections to the computation server.

        Args:
            data: points as numpy array [[x1, y1] [x2, y2]]
        """
        args = {"reset": self._reset_flag,
                "metrics": True}

        self._client.send_array(args, np.array(data))

        # Set _send_flag to True for synchronization
        self._send_flag = True

    @property
    def recv(self) -> Union[List, None]:
        """
        Receive tracking information.

        Returns:
            None if _send_flag is not set, otherwise
            Dict:
                "tracks" : []
                "metrics" : []
        """

        result = self._client.recv_dict()
        output = {}

        if "tracks" in result:
            points = result["tracks"]
            # Update tracks dictionary
            for key, point in points.items():
                if key in self.tracks:
                    self.tracks[key].append(point)
                else:
                    self.tracks[key] = [point]
            output["tracks"] = self.tracks

        # Update server metrics
        self._server_metrics = result["metrics"]

        # Set _send_flag as False
        self._send_flag = False

        # Set _reset_flag as False
        self._reset_flag = False

        return output

    def server_metrics(self) -> dict:
        """
        Returns server metrics.

        Returns:
            output: Server metrics as dictionary
                ["recv"]: Receive time mean
                ["comp"]: Computation time mean
                ["send"]: Sending time mean
                ["load"]: Computation Time / Overall Time
        """
        output = {"recv": self._server_metrics["recv_time"] / self._server_metrics["iteration"],
                  "comp": self._server_metrics["comp_time"] / self._server_metrics["iteration"],
                  "send": self._server_metrics["send_time"] / self._server_metrics["iteration"],
                  "load": self._server_metrics["comp_time"] / self._server_metrics["overall_time"]}

        return output

    def server_summary(self) -> str:
        """
        Returns a short summary of server statistics.

        Returns:
            summary: Returns load and computation time
        """

        metrics = self.server_metrics()
        summary = f"Track Processing Server | Load: {metrics['load']:.3f} | Comp: {metrics['comp']:.3f}   "
        return summary

    def stop(self):
        """Stop and terminate all used sockets"""
        # If there is any waiting recv, just receive it
        if self._send_flag:
            self.recv()

        # Send to the server stop request
        args = {"stop": True}
        self._client.send_array(args, np.ndarray([]))

        # Close the connection
        self._client.close()

        # Terminate the server process
        self._server.terminate()

    def reset_tracks(self):
        """If frame idx changed manually reset tracks"""
        self._reset_flag = True
        self.tracks = {}
