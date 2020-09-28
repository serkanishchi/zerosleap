import numpy as np
from zerosleap.conn.pair import PairClient, PairServer


def test_pair_connection():

    # Initialize the client
    client = PairClient(address=f"tcp://127.0.0.1:5555")

    # Initialize the server
    server = PairServer(f'tcp://*:5555')

    dict_data = {"test_data": True}
    client.send_dict(dict_data)
    assert dict_data == server.recv_dict()
