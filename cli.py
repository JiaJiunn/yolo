import argparse

from detect import detect_img, detect_vid


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

    parser.add_argument("--video", dest="video",
                        help="video file to run detection on", default=None, type=str)
    parser.add_argument('--webcam', dest="webcam", action='store_true')

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

    parser.add_argument("--verbose", dest='verbose', help="1 to print model performance, 0 otherwise",
                        default=0, type=int)

    return parser.parse_args()


def cli_handler():
    args = arg_parse()
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    if args.webcam:
        # run on webcam
        detect_vid(0, args.det, batch_size, confidence,
                   nms_thesh, args.cfgfile, args.weightsfile, args.namesfile, args.reso, args.verbose)

    elif args.video == None:
        # run on images
        detect_img(args.images, args.det, batch_size, confidence,
                   nms_thesh, args.cfgfile, args.weightsfile, args.namesfile, args.reso, args.verbose)
    else:
        # run on video
        detect_vid(args.video, args.det, batch_size, confidence,
                   nms_thesh, args.cfgfile, args.weightsfile, args.namesfile, args.reso, args.verbose)
