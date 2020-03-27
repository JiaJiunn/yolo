import torch
from torch.autograd import Variable
import cv2 as cv
import argparse
import time
import random
import os
import os.path as osp
import pickle as pkl
import pandas as pd

from darknet import Darknet
from utils import write_results, prep_image, load_classes, IMG_DIM


def arg_parse():
    """
    Parse arguements.
    """
    parser = argparse.ArgumentParser(description='YOLO v3')

    # arguments
    parser.add_argument("--video", dest="videofile",
                        help="video file to run detection on", default="video.avi", type=str)

    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)

    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weights/yolov3.weights", type=str)

    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    # for testing, we use weights trained on the COCO dataset
    num_classes = 80
    classes = load_classes("data/coco.names")

    # set up nn
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)

    # parse resolution
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # set the model in evaluation mode
    model.eval()

    def write(x, results):
        """
        Draws bounding boxes.
        """
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results
        cls = int(x[-1])
        color = random.choice(colors)
        label = "{0}".format(classes[cls])
        cv.rectangle(img, c1, c2, color, 1)
        t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv.rectangle(img, c1, c2, color, -1)
        cv.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                   cv.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    # EVALUATION PHASE

    videofile = args.videofile
    cap = cv.VideoCapture(0)  # NOTE videofile = 0 for webcam

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0

    start = time.time()
    # NOTE no need to deal with batches because can only 1 frame at a time
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            img = prep_image(frame, inp_dim)

            # TODO idk
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            # predictions
            with torch.no_grad():
                output = model(Variable(img), CUDA)
            output = write_results(
                output, confidence, num_classes, nms_conf=nms_thesh)

            # TODO still dk what this is compared to below
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.4f}".format(
                    frames / (time.time() - start)))
                cv.imshow("frame", frame)
                key = cv.waitKey(1)

                # quit if user presses Q
                if key & 0xFF == ord('q'):
                    break
                continue

            # transform bounding boxes back to original image dims
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(416/im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor *
                                  im_dim[:, 0].view(-1, 1))/2
            output[:, [2, 4]] -= (inp_dim - scaling_factor *
                                  im_dim[:, 1].view(-1, 1))/2

            output[:, 1:5] /= scaling_factor

            # clip bb outside image to edge of image
            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(
                    output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(
                    output[i, [2, 4]], 0.0, im_dim[i, 1])

            # load classes and color options
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            # draws bb in place
            list(map(lambda x: write(x, frame), output))

            # displays processed image
            cv.imshow("frame", frame)

            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            frames += 1

            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format(
                frames / (time.time() - start)))

        else:
            break
