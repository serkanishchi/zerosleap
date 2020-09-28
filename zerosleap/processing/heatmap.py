"""
Heatmap image processing module.
"""

import numpy as np
import tensorflow as tf


def heatmap_to_rgba(heatmap: np.ndarray, alpha: float = 1.0, color: list = [200, 0, 200]) -> np.ndarray:
    """
    Converts 1 channel heatmap to 4 channel rgba image.

    Args:
        heatmap: 2d np.ndarray [w, h]
        alpha: Opacity value
        color: Color to paint heatmap

    Returns:
        Return heatmap as rgba image (np.ndarray) with 4 channels
    """

    img_a = (heatmap * 255 * alpha).astype(np.uint8)
    img_r = (heatmap * color[0]).astype(np.uint8)
    img_g = (heatmap * color[1]).astype(np.uint8)
    img_b = (heatmap * color[2]).astype(np.uint8)

    # Stack all channels and return
    return np.dstack((img_r, img_g, img_b, img_a))

def find_heatmap_peaks(img: tf.Tensor):
    """
    Find local peaks with selecting local peak if it is greater than its
    8 neighbors. There can be two possible solution for this peak finding.
    The first one is performing 2D Grayscale Dilation or Erosion (with negative
    input) as discussed in the paper and the other one is making max_pooling
    and unpooling with 3 stride, but there is no official unpooling function
    in TensorFlow library. The other pool max solutions other than 3 stride
    not finds the optimal solution.

    Args:
        img: Takes heatmap as an input [b, w, h, c]

    Returns:
        peak_points: peak points [n, [x, y]]
        peak_values: peak confidence values [n, c]
        sample_indexes: peak sample indices [n, p]
    """

    # Perform local peak finding with 2D Grayscale Dilation or Erosion
    max_img = tf.nn.dilation2d(
        input=img,
        filters=tf.reshape([[0., 0., 0.], [0., -1., 0.], [0., 0., 0.]], (3, 3, 1)),
        strides=[1, 1, 1, 1],
        padding="SAME",
        data_format="NHWC",
        dilations=[1, 1, 1, 1]
    )

    # Filter for maxima and threshold.
    max_img = (img > max_img) & (img > 0.2)

    # Return elements where max_img is True
    elements = tf.where(max_img)

    # Take elements as point.
    peak_points = tf.cast(tf.gather(elements, [2, 1], axis=1), tf.float32)

    # Get peak values.
    peak_values = tf.gather_nd(img, elements)

    # Pull out sample indexes.
    sample_indexes = tf.gather(elements, 0, axis=1)

    return peak_points, peak_values, sample_indexes
