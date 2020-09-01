from typing import Dict, List, Tuple

import tensorflow as tf
import numpy as np
import cv2

from config import parse_cfg, _LEAKY_RELU_ALPHA, _IMG_DIM


class DarkNet:
    """Backbone for YOLOv3."""

    def __init__(self, cfg_file: str) -> None:

        self.blocks = parse_cfg(cfg_file)
        # output feature map cache from previous layers, used by route and shortcut layers
        self.outputs = {}
        self.detections = []

    def build(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Returns a list of detection outputs from each scale."""

        # [net] block describes the network input and training params
        self.net_info = self.blocks[0]

        detect_count = 0
        # sequentially process block by block
        for index, block in enumerate(self.blocks[1:]):

            # the five block types

            if block["type"] == "convolutional":
                input_tensor = self._parse_conv_block(input_tensor, block)

            elif block["type"] == "upsample":
                input_tensor = self._parse_upsample_block(input_tensor)

            elif block["type"] == "route":
                input_tensor = self._parse_route_block(
                    input_tensor, block, index)

            elif block["type"] == "shortcut":
                input_tensor = self._parse_shortcut_block(
                    input_tensor, block, index)

            elif block["type"] == "yolo":  # 3 detection layers
                input_tensor = self._parse_yolo_block(input_tensor, block)
                num_classes = int(block["classes"])
                input_tensor = tf.reshape(input_tensor, (-1, 5 + num_classes))
                input_tensor = tf.identity(
                    input_tensor, name=f"output_{detect_count}")
                detect_count += 1

                self.detections.append(input_tensor)

            # update feature map output cache
            self.outputs[index] = input_tensor

        detections = tf.concat(self.detections, axis=0)
        return detections

    def _parse_conv_block(self, input_tensor: tf.Tensor, block: Dict[str, str]) -> tf.Tensor:

        # last layer does not have batch_normalize
        try:
            batch_normalize = int(block["batch_normalize"])
            bias = False
        except:
            batch_normalize = 0
            bias = True

        filters = int(block["filters"])
        stride = int(block["stride"])
        kernel_size = int(block["size"])

        # boolean for whether we should pad the input
        padding = int(block["pad"])
        if padding and stride != 1:
            pad = (kernel_size - 1) // 2
            paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
            input_tensor = tf.pad(input_tensor, paddings, mode='CONSTANT')

        input_tensor = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding=(
            'SAME' if stride == 1 else 'VALID'), use_bias=bias)(input_tensor)

        # batch norm layer
        if batch_normalize:
            input_tensor = tf.keras.layers.BatchNormalization()(input_tensor)

        # activation is either linear or leaky relu for YOLO
        activation = block["activation"]
        if activation == "leaky":
            input_tensor = tf.nn.leaky_relu(
                input_tensor, alpha=_LEAKY_RELU_ALPHA)

        return input_tensor

    def _parse_upsample_block(self, input_tensor: tf.Tensor) -> tf.Tensor:

        input_shape = tf.shape(input_tensor)
        input_tensor = tf.image.resize(
            input_tensor, (input_shape[1] * 2, input_shape[2] * 2))
        return input_tensor

    def _parse_route_block(self, input_tensor: tf.Tensor, block: Dict[str, str], index: int) -> tf.Tensor:

        layers = block["layers"]
        layers = layers.split(',')
        # may have one of two values (start, and maybe end)
        layers = [int(a) for a in layers]

        # either absolute or relative index
        if (layers[0]) > 0:
            layers[0] = layers[0] - index

        # 1 value route
        if len(layers) == 1:
            input_tensor = self.outputs[index + (layers[0])]

        # 2 value route
        else:
            if (layers[1]) > 0:
                layers[1] = layers[1] - index

            fmap_1 = self.outputs[index + layers[0]]
            fmap_2 = self.outputs[index + layers[1]]
            input_tensor = tf.concat([fmap_1, fmap_2], axis=3)

        return input_tensor

    def _parse_shortcut_block(self, input_tensor: tf.Tensor, block: Dict[str, str], index: int) -> tf.Tensor:

        # skip connection
        skip_from = int(block["from"])
        input_tensor = self.outputs[index - 1] + \
            self.outputs[index + skip_from]

        # always linear activation
        return input_tensor

    def _parse_yolo_block(self, input_tensor: tf.Tensor, block: Dict[str, str]) -> tf.Tensor:

        # parse the defined anchors
        mask = block["mask"].split(",")
        mask = [int(x) for x in mask]

        anchors = block["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i + 1])
                   for i in range(0, len(anchors), 2)]
        # tuple of three anchors defined in config
        anchors = [anchors[i] for i in mask]

        # transform predictions
        input_dim = int(self.net_info["height"])
        num_classes = int(block["classes"])

        return self._transform_predictions(input_tensor, anchors, input_dim, num_classes)

    def _transform_predictions(self, prediction: tf.Tensor, anchors: List[Tuple[int, int]], input_dim: int, num_classes: int) -> tf.Tensor:

        conv_shape = tf.shape(prediction)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        stride = input_dim // conv_shape[2]
        anchor_per_scale = len(anchors)  # TODO

        prediction = tf.reshape(
            prediction, (batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes))

        conv_raw_dxdy = prediction[:, :, :, :, 0:2]
        conv_raw_dwdh = prediction[:, :, :, :, 2:4]
        conv_raw_conf = prediction[:, :, :, :, 4:5]
        conv_raw_prob = prediction[:, :, :, :, 5:]

        # sets up grid base coordinates
        y = tf.tile(tf.range(output_size, dtype=tf.int32)
                    [:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)
                    [tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat(
            [x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [
                          batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # gets transformed predictions
        stride = tf.cast(stride, tf.float32)
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors)  # * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
