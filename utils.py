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


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    Takes detection output of shape B x 22743 x 85, and apply
    objectness score thresholding and non max suppression.
    [confidence] is the objectness score threshold, and 
    [nms_conf] is the nms iou threshold.
    """

    # idk what this is
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # convert (center_x, center_y, height, width) to
    # (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # NOTE each input image may have different num true detections, so must be
    # done one image a time (can't vectorize)

    # we loop through indices of images in a batch
    batch_size = prediction.size(0)

    # again, write for init output tensor for true detections
    write = False
    for ind in range(batch_size):

        # get image tensor
        image_pred = prediction[ind]

        # object confidence thresholding

        # get max score class and its score (instead of scores for all 80 classes)
        # TODO details
        max_conf, max_conf_score = torch.max(
            image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # TODO we have set the bb rows with object conf less than threshold to 0
        # now we get rid
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))

        # for cases with no detections (continue skips image)
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # now get true detection classes (-1 is the class index)
        img_classes = unique(image_pred_[:, -1])

        # NMS class-wise

        for cls in img_classes:
            pass


def unique(tensor):

    # TODO understand
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
