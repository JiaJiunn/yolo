import argparse

from detect import detect_img


def arg_parse():
    """
    Parse arguements.
    """
    parser = argparse.ArgumentParser(description='YOLO v3')

    # arguments
    parser.add_argument("--image", dest='image', help="image to perform detection upon",
                        default="imgs/dog_cycle_car.png", type=str)

    parser.add_argument("--cfg", dest="cfg_file", help="config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest="weights_dir", help="weights directory",
                        default="weights", type=str)
    parser.add_argument("--names", dest="names_file", help="dataset names file",
                        default="data/coco.names", type=str)

    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS threshhold", default=0.5)
    parser.add_argument("--iou_thresh", dest="iou_thresh",
                        help="IOU threshhold", default=0.4)

    return parser.parse_args()


def cli_handler():
    args = arg_parse()
    nms_thresh = float(args.nms_thresh)
    iou_thresh = float(args.iou_thresh)

    detect_img(args.image, args.cfg_file, args.weights_dir,
               args.names_file, nms_thresh, iou_thresh)
