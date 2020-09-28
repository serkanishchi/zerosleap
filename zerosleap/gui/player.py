"""
Video player widget for showing video and other informations
on the video.
"""

from PySide2 import QtGui
from PySide2.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QStyle, QSlider, QHBoxLayout, \
    QFileDialog, QSizePolicy
from PySide2.QtGui import QPixmap, QFont, QIcon, QPainter, QColor
from PySide2.QtCore import Qt, QSize

from zerosleap.gui.utils import get_random_color, convert_cv_qt
from zerosleap.gui.controller import VideoController


class VideoPlayer(QWidget):
    """
    Main widget for displaying video, heatmaps, peaks, tracks and other
    information.
    """
    def __init__(self, main_window):
        super().__init__()

        self.disply_width = 1000
        self.display_height = 800
        self.main_window = main_window

        btn_size = QSize(16, 16)

        # Open button for opening video files
        self.open_button = QPushButton("Open Video")
        self.open_button.setToolTip("Open Video File")
        self.open_button.setStatusTip("Open Video File")
        self.open_button.setFixedHeight(24)
        self.open_button.setIconSize(btn_size)
        self.open_button.setFont(QFont("Helvetica", 8))
        self.open_button.setIcon(QIcon.fromTheme("edit-undo"))
        self.open_button.clicked.connect(self.open_video_file)

        # Label for drawing scenes
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Play button
        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setFixedHeight(24)
        self.play_button.setIconSize(btn_size)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play)

        # Pause button
        self.pause_button = QPushButton()
        self.pause_button.setEnabled(True)
        self.pause_button.setFixedHeight(24)
        self.pause_button.setIconSize(btn_size)
        self.pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_button.clicked.connect(self.pause)

        # Next button
        self.next_button = QPushButton()
        self.next_button.setEnabled(False)
        self.next_button.setFixedHeight(24)
        self.next_button.setIconSize(btn_size)
        self.next_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.next_button.clicked.connect(self.next)

        # Back button
        self.back_button = QPushButton()
        self.back_button.setEnabled(False)
        self.back_button.setFixedHeight(24)
        self.back_button.setIconSize(btn_size)
        self.back_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.back_button.clicked.connect(self.back)

        # Seeker
        self.seeker = QSlider(Qt.Horizontal)
        self.seeker.setRange(0, 5000)
        self.seeker.valueChanged.connect(self.seeker_value_changed)
        self.seeker.sliderMoved.connect(self.seeker_slider_moved)

        # Layout for media player controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.back_button)
        control_layout.addWidget(self.seeker)

        # Vertical layout for scene (image_label) and player controls
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(control_layout)

        self.setLayout(layout)

    def open_video_file(self):
        """Shows dialog for opening a video file"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Select the media.",
                                                   ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if file_name != '':
            # If video already loaded restart worker
            if hasattr(self, "_video_controller"):
                self._video_controller.stop()
                self._video_controller.wait()

            self._video_controller = VideoController(file_name)

            # Connect video controller signals to functions
            self._video_controller.scene_changed.connect(self.update_scene)
            self._video_controller.tracks_updated.connect(self.main_window.update_track_viewer)
            self._video_controller.player_paused.connect(self.pause)
            self._video_controller.seeker_range_updated.connect(self.seeker_range_update)
            self._video_controller.seeker_value_updated.connect(self.seeker_value_update)
            self._video_controller.server_stats_updated.connect(self.main_window.update_server_stats)

            # Start the thread
            self._video_controller.start()

            # Reset track viewer
            self.main_window.reset_tracks_table()

            # Set controllers state as play
            self.play_mode(play=True)

    def closeEvent(self, event) -> None:
        # If has _video_controller close it.
        if hasattr(self, "_video_controller"):
            self._video_controller.stop()
        event.accept()

    def seeker_value_changed(self) -> None:
        pass

    def seeker_range_update(self, max_range) -> None:
        """
        Updates slider range according to the input video.
        Seeker range updated whenever a new video file opened.

        Args:
            max_range: Maximum value of slider. (0-max)
        """
        self.seeker.setRange(0, max_range)

    def seeker_value_update(self, value):
        """
        Sets slider value. Updates for each frame

        Args:
            value: New value of the slider
        """
        self.seeker.setValue(value)

    def seeker_slider_moved(self, p):
        """
        Changes controller active frame number if slider moved

        Args:
            p: Slider active point
        """
        self._video_controller.set_frame(p)

    def play(self):
        """Play button action"""
        if hasattr(self, "_video_controller"):
            self._video_controller._paused = False
            self.play_mode(play=True)

    def pause(self):
        """Pause button action"""
        if hasattr(self, "_video_controller"):
            self._video_controller._paused = True
            self.play_mode(play=False)

    def next(self):
        """Next button action"""
        if hasattr(self, "_video_controller"):
            self._video_controller.set_frame(self.seeker.value() + 1)

    def back(self):
        """Next button action"""
        if hasattr(self, "_video_controller"):
            self._video_controller.set_frame(self.seeker.value() - 1)

    # Changes video player controls according to the video player status
    def play_mode(self, play=True):
        """
        Changes video player controls according to the video player status

        Args:
            play: Status of the media player
        """
        self.play_button.setEnabled(not play)
        self.pause_button.setEnabled(play)
        self.next_button.setEnabled(not play)
        self.back_button.setEnabled(not play)

    def update_scene(self, meta_frame):
        """
        Updates the image_label with a new opencv image.

        Args:
            meta_frame: MetaFrame Object
        """

        # Get frame and other data from meta_frame
        frame = meta_frame.frame
        peaks = meta_frame.peaks
        tracks = meta_frame.tracks

        # Get display_options from main_window
        display_options = self.main_window.display_options

        # Get display dimensions
        self.display_height = self.image_label.height()
        self.display_width = self.image_label.width()

        # Adjust the scale factor
        scale_factor = self.display_height / 256

        # Convert frame to qt image
        pm_img = convert_cv_qt(frame, self.display_width, self.display_height)

        # Start painter
        painter = QPainter()
        painter.begin(pm_img)

        # Create pen
        pen = QtGui.QPen()
        pen.setWidth(2)
        pen.setColor(QtGui.QColor(204, 0, 0))  # r, g, b
        painter.setPen(pen)

        if display_options["show_tracks"] == True:
            # Plot peaks with centered
            for point in peaks:
                painter.drawEllipse((point[0] * scale_factor)-4,
                                    (point[1] * scale_factor)-4, 8, 8)

        pen.setWidth(2)
        pen.setColor(QtGui.QColor(0, 0, 255))

        if display_options["show_labels"] == True:
            # Draw tracked objects labels
            for label, trace in tracks.items():
                if len(trace) > 1:
                    # Assign random color for different tracks.
                    color = get_random_color(int(label))
                    qcolor = QColor()
                    qcolor.setNamedColor(color)
                    pen.setColor(qcolor)
                    painter.setPen(pen)
                    label = label
                    label_pos_x = trace[-1][0][0] * scale_factor + 10
                    label_pos_y = trace[-1][0][1] * scale_factor + 10
                    painter.drawText(label_pos_x, label_pos_y, label)

        if display_options["show_traces"] == True:
            # For identified object tracks draw tracking line
            # Use various colors to indicate different track_id
            for label, trace in tracks.items():
                if len(trace) > 1:
                    # Assign random color for different tracks.
                    color = get_random_color(int(label))
                    qcolor = QColor()
                    qcolor.setNamedColor(color)
                    pen.setColor(qcolor)
                    painter.setPen(pen)
                    limit = 0
                    if len(trace) > 200:
                        limit = len(trace) - 200
                    for j in range(limit, len(trace) - 1):
                        # Draw trace line
                        x1 = trace[j][0][0] * scale_factor
                        y1 = trace[j][0][1] * scale_factor
                        x2 = trace[j + 1][0][0] * scale_factor
                        y2 = trace[j + 1][0][1] * scale_factor
                        painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        # painter.drawImage(0, 0, pix_map)
        painter.end()
        # label = QLabel()
        self.image_label.setPixmap(QPixmap.fromImage(pm_img))
