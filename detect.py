import tensorflow as tf
import numpy as np
import pickle as pkl
import random
import cv2

from darknet import DarkNet, _IMG_DIM
from utils import get_bb_boxes, non_max_suppression
from weights_to_ckpt import load_coco_names


def detect_img(image: str, cfg_file: str, weights_dir: str, names_file: str, nms_thresh: float, iou_thresh: float) -> None:

    img = cv2.imread(image)
    img_shape = img.shape
    img = cv2.resize(img, (_IMG_DIM, _IMG_DIM))
    img_ = img[np.newaxis, :, :, :]/255.0

    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:

        # restore weights
        weights_file = weights_dir + '/model.ckpt.meta'
        saver = tf.compat.v1.train.import_meta_graph(weights_file)
        saver.restore(sess, tf.train.latest_checkpoint(weights_dir + '/'))

        # access input and output tensors
        output_0 = graph.get_tensor_by_name("detector/output_0:0")
        output_0 = get_bb_boxes(output_0)
        output_1 = graph.get_tensor_by_name("detector/output_1:0")
        output_1 = get_bb_boxes(output_1)
        output_2 = graph.get_tensor_by_name("detector/output_2:0")
        output_2 = get_bb_boxes(output_2)
        inputs = graph.get_tensor_by_name("inputs:0")

        # run inference
        detected_boxes_0, detected_boxes_1, detected_boxes_2 = sess.run(
            [output_0, output_1, output_2], feed_dict={inputs: img_})
        detected_boxes = np.concatenate(
            [detected_boxes_0, detected_boxes_1, detected_boxes_2], axis=1)

        # post processing
        filtered_boxes = non_max_suppression(
            detected_boxes, nms_thresh, iou_thresh)

        # draw image
        classes = load_coco_names(names_file)
        colors = pkl.load(open("pallete", "rb"))
        for cls_idx, bboxs in filtered_boxes.items():
            color = random.choice(colors)
            for box, score in bboxs:
                c1 = tuple(int(x) for x in box[0:2])
                c2 = tuple(int(x) for x in box[2:])
                cv2.rectangle(img, c1, c2, color, 1)
                label = '{} {:.2f}%'.format(classes[cls_idx], score * 100)
                t_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.putText(
                    img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 0], 1)
        img = cv2.resize(img, (img_shape[1], img_shape[0]))
        cv2.imwrite('output.png', img)
