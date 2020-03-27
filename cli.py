import argparse

from detect import detect


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
    parser.add_argument("--reso", dest='reso', help="input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="608", type=str)

    parser.add_argument("--cfg", dest='cfgfile', help="config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weights file",
                        default="weights/yolov3.weights", type=str)
    parser.add_argument("--names", dest='namesfile', help="dataset names file",
                        default="data/coco.names", type=str)

    return parser.parse_args()


def cli_handler():
    args = arg_parse()
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    detect(args.images, args.det, batch_size, confidence,
           nms_thesh, args.cfgfile, args.weightsfile, args.namesfile, args.reso)
