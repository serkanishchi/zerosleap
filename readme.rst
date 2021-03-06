ZeroSleap
=========================================

.. image:: resources/app.png
    :width: 800px

|

Task A: Build s server that can interactively perform model inference.

- I have developed the computation module for parallel processing. This module using ZeroMQ for messaging between processes. VideoProcessingServer and TrackProcessingServer implemented in the context of this module.

- The application is developed using PySide2 library and it supports user interactions like video seeking.

- Heatmaps are processed to extract peaks using local peak finding and displayed at the user interface.

Task B: Implement a multi-object tracker to assign the predicted centroids to the correct animal over time.

- Kalman Filter is used to track detections and these detections assigned to the objects over time with label and trace information.

- VideoProcessing (inference + peak finding) and TrackProcessing statistics calculated and displayed at the status bar in realtime.

- Tracking algorithm implemented in parallel.

Installation and Setup
=========================================
Clone the repository.

- git clone https://github.com/serkanishchi/zerosleap.git

- cd zerosleap

Create Virtual Env.

- python3.8 -m venv env

Activate.

- source env/bin/activate

Install requirements.

- pip install -r requirements.txt

Setup package.

- python setup.py install

Download Pretrained Model.

- https://www.dropbox.com/s/kuyqdopree4fh0r/best_model.h5?dl=1

- Put the model under the "zerosleap/resources" folder with name of keras_model.h5

Run the application.

- python zerosleap/main.py

