"""
GUI Main Window for displaying video and processing results like
heatmaps, peaks and tracks.

Main Window has a VideoPlayer widget for reading, controlling
and processing video files.

Main Window also has a QTableView object to show tracks information
at each frame.

Also there is a toolbar for filtering display options and status
bar for displaying comparing Video Processing Server and Track \
Processing Server statistics.
"""

from PySide2.QtWidgets import QWidget, QLabel, QStyle, QStatusBar, \
    QHBoxLayout, QMainWindow, QTableView, QToolBar, QAction, QCheckBox, QMessageBox
from PySide2.QtGui import QStandardItemModel, Qt, QStandardItem, QBrush, QColor

from zerosleap.gui.utils import get_random_color
from zerosleap.gui.player import VideoPlayer


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        layout = QHBoxLayout()

        # video player object
        self.video_player = VideoPlayer(self)

        # Tracks table
        self.tracks_table = QTableView()

        # Keeps display options
        self.display_options = {"show_tracks": True,
                                "show_traces": True,
                                "show_labels": True}

        layout.addWidget(self.video_player)
        layout.addWidget(self.tracks_table)

        # Create tracks model and items to show in tracks table.
        self.tracks_model = QStandardItemModel()
        self.tracks_model.setHorizontalHeaderLabels(["Id", "State", "Age"])
        self.tracks_table.setModel(self.tracks_model)
        self.track_items = {}

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.setMinimumHeight(800)
        self.setMinimumWidth(1000)

        self.tracks_table.setFixedWidth(300)

        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        open_button = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Open Video File", self)
        open_button.setStatusTip("Open Video File")
        open_button.triggered.connect(self.video_player.open_video_file)
        toolbar.addAction(open_button)

        toolbar.addWidget(QLabel("Tracks"))
        self.tracks_check_box = QCheckBox()
        self.tracks_check_box.toggle()
        self.tracks_check_box.stateChanged.connect(self.tracks)
        toolbar.addWidget(self.tracks_check_box)

        toolbar.addWidget(QLabel("Traces"))
        self.traces_check_box = QCheckBox()
        self.traces_check_box.toggle()
        self.traces_check_box.stateChanged.connect(self.traces)
        toolbar.addWidget(self.traces_check_box)

        toolbar.addWidget(QLabel("Labels"))
        self.labels_check_box = QCheckBox()
        self.labels_check_box.toggle()
        self.labels_check_box.stateChanged.connect(self.labels)
        toolbar.addWidget(self.labels_check_box)

        self.status_bar = QStatusBar()
        self.vps_server_label = QLabel("")
        self.tps_server_label = QLabel("")

        self.status_bar.addWidget(self.vps_server_label)
        self.status_bar.addWidget(self.tps_server_label)

        self.setStatusBar(self.status_bar)

    def get_or_create_track_row(self, track: dict) -> ():
        """
        Get or create a track row in tracks table

        Args:
            track: Track label and trace
        """

        track_id = track[0]
        if track_id not in self.track_items:
            self.track_items[track_id] = self.add_track_row(track)
        return self.track_items[track_id]

    def add_track_row(self, track):
        """
        Adds a track row to the tracks table model.

        Args:
            track: Track label and trace

        Returns:
            id_item
            status_item
            age_item
        """
        id_item = QStandardItem()
        id_item.setText(track[0])
        id_item.setBackground(QBrush(QColor(
            get_random_color(int(track[0]))
        )))
        id_item.setEditable(False)

        status_item = QStandardItem()
        status_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        status_item.setEditable(False)

        age_item = QStandardItem()
        age_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        age_item.setEditable(False)

        self.tracks_model.appendRow([id_item, status_item, age_item])
        self.tracks_model.sort(0)
        return id_item, status_item, age_item

    def update_track_row(self, track_id, trace):
        """
        Updates track list and model

        Args:
            track_id
            trace
        """
        status = "Active"
        if track_id in self.track_items:
            # If length of the track trace is not changed
            # Track is passive
            if len(self.track_items[track_id]) == len(trace):
                status = "Passive"
        track_id, track_status, track_age = self.get_or_create_track_row(track_id)
        track_status.setText(status)
        track_age.setText(f"{len(trace)//24}")

    def reset_tracks_table(self):
        """Reset tracks_model and track_items"""
        self.tracks_model.removeRows(0, len(self.track_items))
        self.track_items = {}

    def update_server_stats(self, vps_stats_summary, tps_stats_summary):
        """
        Updates server statistics.

        Args:
            vps_stats_summary: Video Processing Server statistics
            tps_stats_summary: Track Processing Server statistics
        """
        self.tps_server_label.setText(tps_stats_summary)
        self.vps_server_label.setText(vps_stats_summary)

    def update_track_viewer(self, tracks):
        """
        Update tracks table

        Args:
            tracks
        """
        if not tracks:  # Skip update if we have no data.
            return

        for track_id, trace in tracks.items():
            self.update_track_row(track_id, trace)

    def traces(self):
        """Changes trace display options."""
        self.display_options["show_traces"] = not self.display_options["show_traces"]

    def labels(self):
        """Changes labels display options."""
        self.display_options["show_labels"] = not self.display_options["show_labels"]

    def tracks(self):
        """Changes tracks display options."""
        self.display_options["show_tracks"] = not self.display_options["show_tracks"]

    def closeEvent(self, event):
        """Exit application."""
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.video_player.close()
            event.accept()

        else:
            event.ignore()
