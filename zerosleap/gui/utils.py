import cv2
import numpy as np
from PySide2 import QtGui
from PySide2.QtCore import Qt
from PySide2.QtGui import QImage


def get_random_color(obj_id):
    """Defines different and stable color for each obj_id"""

    colors = [
        '#FF0000', '#FFFF00', '#00FF00', '#0000FF', '#00FFFF', '#FF00FF', '#FFA500',
        '#AFEEEE', '#8B008B', '#FF69B4', '#FFFACD', '#DEB887', '#F0FFF0', '#B0C4DE',
    ]

    return colors[obj_id % len(colors)]


def hhmmss(frame, fps=24):
    """Transform number of the frames to "hhmmss" format with given frame per second"""

    s = round(frame / fps)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))


def convert_cv_qt(cv_img: np.ndarray, display_width: int, display_height: int) -> QImage:
    """Convert from an opencv image to QPixmap"""

    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimage = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_ARGB32)
    qimage_scaled = qimage.scaled(display_width, display_height, Qt.KeepAspectRatio)
    return qimage_scaled
