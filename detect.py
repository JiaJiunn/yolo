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
from utils import write_results, prep_image, IMG_DIM


def arg_parse():
    """
    Parse arguements.
    """
    parser = argparse.ArgumentParser(description='YOLO v3')

    # arguments
    parser.add_argument("--images", dest='images', help="image/directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="image/directory to store detections to",
                        default="det", type=str)

    parser.add_argument("--bs", dest="bs", help="batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="object confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS threshhold", default=0.4)

    parser.add_argument("--cfg", dest='cfgfile', help="config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weights file",
                        default="weights/yolov3.weights", type=str)

    parser.add_argument("--reso", dest='reso', help="input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="608", type=str)

    return parser.parse_args()


def load_classes(names_file):
    """
    Returns a dictionary of index-class names listed in [names_file].
    """
    fp = open(names_file, "r")
    names = fp.read().split("\n")[:-1]
    return names


if __name__ == "__main__":
    args = arg_parse()
    images = args.images
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

    # EVALUATION PHASE

    # for time keeping 1
    read_dir = time.time()

    # read input images
    try:
        imlist = [osp.join(osp.realpath('.'), images, img)
                  for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    # for output images
    if not os.path.exists(args.det):
        os.makedirs(args.det)

    # for time keeping 2
    load_batch = time.time()

    # use opencv to read images
    loaded_ims = [cv.imread(x) for x in imlist]

    # NOTE opencv returns images as np array with format BGR
    # hence we use prep image to convert the format
    # returns PyTorch Variables for images
    im_batches = list(map(prep_image, loaded_ims, [
                      inp_dim for x in range(len(imlist))]))

    # list containing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    # creates the batches # TODO
    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)*batch_size,
                                                              len(im_batches))])) for i in range(num_batches)]

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    # detection loop

    write = 0
    start_det_loop = time.time()  # for time keeping 3
    for i, batch in enumerate(im_batches):

        start = time.time()  # for time keeping 4

        if CUDA:
            batch = batch.cuda()

        # run detection
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        prediction = write_results(
            prediction, confidence, num_classes, nms_conf=nms_thesh)

        end = time.time()  # for time keeping 5

        # idk what this is doing
        if type(prediction) == int:
            for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                print("{0:20s} predicted in {1:6.3f} seconds".format(
                    image.split("/")[-1], (end - start)/batch_size))
                print("{0:20s} {1:s}".format("Objects Detected:", ""))
                print("----------------------------------------------------------")
            continue

        # transform index of image in batch to index of image in imlist
        prediction[:, 0] += i*batch_size

        # init output
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        # print time taken for each object and object detected
        for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

        if CUDA:
            torch.cuda.synchronize()

    # skip if no objects detected
    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    # draw bounding boxes
    # first need to transform bb to original dim of image (before padding) # TODO
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(IMG_DIM/im_dim_list, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 0].view(-1, 1))/2
    output[:, [2, 4]] -= (inp_dim - scaling_factor *
                          im_dim_list[:, 1].view(-1, 1))/2

    # and also undo the rescaling in letterbox_image
    output[:, 1:5] /= scaling_factor

    # clip bb outside image to edge of image
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(
            output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(
            output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()
    class_load = time.time()  # for time keeping 7

    # pickled class with many colors to choose from
    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()  # for time keeping 8

    def write(x, results):
        """
        Draw bounding boxes.
        """
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])

        # random color for bb
        color = random.choice(colors)

        # class label
        label = "{0}".format(classes[cls])
        cv.rectangle(img, c1, c2, color, 1)
        t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv.rectangle(img, c1, c2, color, -1)
        cv.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                   cv.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        return img

    # draw bb (modifies images in place)
    list(map(lambda x: write(x, loaded_ims), output))

    # save each image by prefixing "det_"
    det_names = pd.Series(imlist).apply(
        lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))
    list(map(cv.imwrite, det_names, loaded_ims))

    end = time.time()  # for time keeping 9

    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format(
        "Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format(
        "Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format(
        "Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format(
        "Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()
