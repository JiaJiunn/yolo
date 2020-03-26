import torch
import torch.nn as nn


def parse_cfg(cfg_file):
    """
    Returns a list of blocks, where each block is represented as a dictionary.

    Note: there are six types of blocks:
    - convolutional
    - shortcut
    - upsample
    - route
    - yolo
    - net
    """

    file = open(cfg_file, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            # marks new block
            if len(block) != 0:
                # append old block
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    """
    Used in route and shortcut layers as a dummy layer. The actual operations
    (concatenation or addition) will be defined only in [forward] of the 
    darknet network.
    """

    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """
    Used in detection layer, keeps track of anchors that will be used to detect
    bounding boxes.
    """

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
    Takes list of blocks from [parse_config] and returns a tuple of:
    - the network info
    - list of modules based on the parsed configurations
    """
    # [net] block describes the network input and training params (not really a block type)
    net_info = blocks[0]

    # to keep track of the depth of kernels of each layer (starts with 3 for RGB)
    prev_filters = 3
    output_filters = []

    # want to return a list of blocks, where each block is a nn.Sequential module
    module_list = nn.ModuleList()

    for index, x in enumerate(blocks[1:]):

        module = nn.Sequential()

        if (x["type"] == "convolutional"):

            # last layer does not have batch_normalize
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            padding = int(x["pad"])
            kernel_size = int(x["size"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            activation = x["activation"]
            filters = int(x["filters"])
            stride = int(x["stride"])

            # TODO understand

            # convolutional layer
            conv = nn.Conv2d(prev_filters, filters,
                             kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # activation: for yolo, it is either linear or leaky relu
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        elif (x["type"] == "upsample"):

            stride = int(x["stride"])

            # for upsampling layer, use bilinear upsampling
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x["type"] == "route"):

            # may have one of two values (start, and maybe end)
            x["layers"] = x["layers"].split(',')

            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # Positive anotation # TODO ???
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            # just use a dummy layer and define operations later in forward
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end < 0:
                filters = output_filters[index +
                                         start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif (x["type"] == "shortcut"):

            # also use a dummy layer and define operations later in forward
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif (x["type"] == "yolo"):

            # yolo is the detection layer (only 3 of them for 3 scales)
            # we keep track of the chosen anchors used to detect bounding boxes

            # choose anchors based on list of masks
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1])
                       for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            # detection layer
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)

        # TODO re-follow logic
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


# test function
def test_parsing_and_module_creation(cfg_file):
    blocks = parse_cfg(cfg_file)
    print(create_modules(blocks))


# if __name__ == "__main__":
#     test_parsing_and_module_creation('cfg/yolov3.cfg')
