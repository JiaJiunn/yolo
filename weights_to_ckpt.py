#! /usr/bin/env python3

import tensorflow.compat.v1 as tf
import numpy as np

from typing import Dict

from darknet import DarkNet, _IMG_DIM


def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    # import pdb; pdb.set_trace()
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(
                        tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr +
                                       bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(
                    tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def load_coco_names(file_name: str) -> Dict[int, str]:
    """Returns dictionary of classes."""

    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def main():

    classes = load_coco_names("data/coco.names")
    tf.disable_v2_behavior()
    inputs = tf.placeholder(
        tf.float32, [None, _IMG_DIM, _IMG_DIM, 3], name="inputs")

    with tf.variable_scope('detector'):
        # build model
        darknet = DarkNet("cfg/yolov3.cfg")
        darknet.build(inputs)
        load_ops = load_weights(tf.global_variables(
            scope='detector'), 'weights/yolov3.weights')

    saver = tf.train.Saver(tf.global_variables(scope='detector'))
    with tf.Session() as sess:
        sess.run(load_ops)
        save_path = saver.save(sess, save_path='./weights/model.ckpt')


if __name__ == "__main__":
    main()
