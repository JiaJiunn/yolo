import torch
import numpy as np


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
    """
    [prediction] is a 3d feature map, which we convert to a 2d tensor of the 
    form: [1st bb (0,0), 2nd bb (0,0), 3rd bb (0,0), 1st bb (0,1), ,,,]
    """
    # set attr
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # TODO idk what this does - copy?
    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # divide anchors by stride
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # sigmoids the centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # TODO im not sure about this
    # add the center offsets
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # log space transform height and the width
    # TODO applies anchors to bounding boxes??
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors

    # sigmoid activation for class scores
    prediction[:, :, 5: 5 +
               num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # resize feature map up back to input image size
    prediction[:, :, :4] *= stride

    return prediction
