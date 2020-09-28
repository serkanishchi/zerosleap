"""
This module provides  a communication pipeline between threads
and processes.

Uses PAIR socket for bidirectional communication. Its a very simple
socket and no need to think whether you have read the complete
message or not. There can only be one connected peer, server
listens on a certain port and client connects to it. Its a low
level socket and should be used carefully between threads.
A dynamic loop that uses flags can provide parallelism
and can prevent possible latencies. This is not the perfect
solution but its easy and meets the requirements.

This can be replaced with SUB/PUB socket using two separate connection
in a node for using sending and receiving.
"""

import zmq
import numpy as np

from zerosleap.conn.serialize import SerializingContext


class PairNode:
    """
    Creates a node for bidirectional communication.
    This object does not create any connection. Should be
    initialized manually or PairClient and PairServer nodes
    should be used instead.
    """

    def __init__(self):
        """
        Initialize the PairNode but not creates any
        connection or binding.
        """
        super().__init__()
        socket_type = zmq.PAIR
        self._zmq_context = SerializingContext()
        self._zmq_socket = self._zmq_context.socket(socket_type)

    def send_array(self, args: dict, data: np.ndarray):
        """
        Serialize and sends ndarray (numpy array) data

        Args:
            args: The parameters we need to send additional to data
            data: Serialized ndarray (numpy array) data

        Returns:
            None
        """

        if data.flags['C_CONTIGUOUS']:
            # if data is already contiguous in memory just send it
            self._zmq_socket.send_array(data, args)
        else:
            # else make it contiguous before sending
            image = np.ascontiguousarray(data)
            self._zmq_socket.send_array(image, args)

    def recv_array(self) -> [dict, np.ndarray]:
        """
        Receive serialized ndarray (numpy array) data with
        non-copy manner. Copy mode is more expensive.

        Returns:
            args: The parameters we need to send additional to data
            data: Serialized ndarray (numpy array) data
        """
        args, data = self._zmq_socket.recv_array()
        return args, data

    def send_dict(self, data: dict):
        """
        Sends already serialized dictionary data. Dictionary
        values also should be serialized.

        Args:
            data: Dictionary data

        Returns:
            None
        """
        self._zmq_socket.send_json(data)

    def recv_dict(self) -> dict:
        """
        Receives serialized dictionary data

        Returns:
            data: Dictionary data
        """
        data = self._zmq_socket.recv_json(flags=0)
        return data

    def close(self):
        """Closes ZMQ Socket and Context"""
        self._zmq_socket.close()
        self._zmq_context.term()

    def __enter__(self):
        """Enables usage of WITH statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Enables usage of WITH statement"""
        self.close()


class PairClient(PairNode):
    """
    Creates a Client node using PairNode
    """

    def __init__(self, address='tcp://127.0.0.1:5555'):
        """
        Initialize the PairClient

        Args:
            address: Server address to be connected
        """
        super(PairClient, self).__init__()
        # Connects to the server
        self._zmq_socket.connect(address)


class PairServer(PairNode):
    """
    Creates a Server node using PairNode
    """

    def __init__(self, address='tcp://*:5555'):
        """
        Initialize the PairServer

        Args:
            address: Client address to be bind
        """
        super(PairServer, self).__init__()
        # Binds the client
        self._zmq_socket.bind(address)
