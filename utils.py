import tensorflow as tf
import numpy as np


def get_bb_boxes(detections: tf.Tensor) -> tf.Tensor:

    center_x, center_y, width, height, attrs = tf.split(
        [detections], [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)
    return detections


def non_max_suppression(predictions_with_boxes: tf.Tensor, confidence_threshold: int, iou_threshold: int) -> tf.Tensor:

    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result


def _iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Computes Intersection over Union value for 2 bounding boxes
    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou
