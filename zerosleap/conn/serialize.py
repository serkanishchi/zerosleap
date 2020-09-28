import zmq
import numpy as np


class SerializingSocket(zmq.Socket):
    """
    Defines a serializing socket for sending and receiving
    ndarray (numpy array) and OpenCV Images
    """

    def send_array(self, data, args=None, flags=0):
        """
        Sends numpy array with metadata necessary for reconstructing
        the array (dtype, shape) and additional arguments.

        Args:
            data: Numpy array or OpenCV image.
            args: The parameters we need to send additional to data
            flags:
        """

        # Serialize the data
        meta_data = {'args': args,
                     'dtype': str(data.dtype),
                     'shape': data.shape}

        # Send the meta data
        self.send_json(meta_data, flags | zmq.SNDMORE)

        # Send the numpy array
        self.send(data, flags, copy=False, track=False)

    def recv_array(self, flags=0):
        """
        Receives numpy array with metadata necessary for reconstructing
        the array (dtype, shape) and additional arguments.

        Args:
            flags: Connection flag

        Returns:
            args: The parameters we need to send additional to data
            data: Numpy array or OpenCV image reconstructed with dtype and shape.
        """

        # Receives meta data for reconstruction
        meta_data = self.recv_json(flags=flags)

        # Receives the serialized data
        serialized_data = self.recv(flags=flags, copy=False, track=False)

        # Reconstruct the data
        data = np.frombuffer(serialized_data, dtype=meta_data['dtype'])

        # Returns reconstructed numpy array
        return meta_data['args'], data.reshape(meta_data['shape'])


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket
