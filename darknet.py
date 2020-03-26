import torch
import torch.nn as nn

from config import parse_cfg, create_modules


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
                pass
