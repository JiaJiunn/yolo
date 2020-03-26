import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 as cv

from config import parse_cfg, create_modules
from utils import predict_transform


class Darknet(nn.Module):

    def __init__(self, cfgfile):

        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        """
        [x] is the input, and
        [CUDA] is true if we want to use GPU to accelerate the forward pass.
        """
        # first block just defines the net
        modules = self.blocks[1:]

        # output feature map cache from previous layers, used by route and shortcut layers
        outputs = {}

        # NOTE for detection layer, we cannot init empty tensor then concat
        # so we use this whack write=0 to check if encountered first tensor
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            # we already used pytorch's pre-built layers for convolutional and upsampling
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":

                layers = module["layers"]
                layers = [int(a) for a in layers]

                # TODO understand this

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                # 1 value route
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                # 2 value route
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)  # dim=1 for cat along depth

            elif module_type == "shortcut":

                # TODO understand this
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':

                # detection layer
                anchors = self.module_list[i][0].anchors

                # get input dimensions and num classes
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                # transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                # initialize collector
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            # update feature map output cache
            outputs[i] = x

        return detections

    def load_weights(self, weights_file):
        """
        Weights are only for convolution blocks only.
        When convolution blocks have batch norm layer, then there are no bias;
        otherwise bias weights need to be read from file.
        """
        fp = open(weights_file, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4&5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)

        # rest of bits now represent weights
        weights = np.fromfile(fp, dtype=np.float32)

        # load weights into modules of network
        # ptr to keep track of where we are in weights array
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # weights are for convolutional blocks only, otherwise ignore
            if module_type == "convolutional":

                # TODO understand conv batch norm structure
                model = self.module_list[i]

                # check if there is a batch norm layer
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    # no bias
                    bn = model[1]

                    # get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # load the weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # TODO i don't fully understand loading strat below - bn is updated?

                    # cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # no batch norm, need to load batch norm bias

                    # number of biases
                    num_biases = conv.bias.numel()

                    # load the weights
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # copy the data
                    conv.bias.data.copy_(conv_biases)

                # load conv weights
                num_weights = conv.weight.numel()

                # cast and copy to model
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

### TESTS ###


def process_test_input(img_file):

    img = cv.imread(img_file)

    # resize to input dim (specified in cfg file)
    img = cv.resize(img, (608, 608))

    # TODO idu this fully
    # BGR -> RGB | H X W C -> C X H X W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))

    # add a channel at 0 (for batch) and normalise
    img_ = img_[np.newaxis, :, :, :]/255.0

    # convert to float then Variable (to perform tensor ops and compute grad)
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    return img_


def test_forward_pass(cfg_file, weight_file, img_file):

    model = Darknet(cfg_file)
    model.load_weights(weight_file)
    input_img = process_test_input(img_file)

    # feed into forward pass
    pred = model(input_img, False)  # torch.cuda.is_available()
    print(pred)
    print(pred.size())
    # 1 batch size
    # 85 per row for 4 bbox attributes, 1 objectness score, and 80 class scores


# test
test_forward_pass('cfg/yolov3.cfg', 'weights/yolov3.weights',
                  'imgs/dog_cycle_car.png')
