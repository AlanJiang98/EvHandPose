"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch


def purge_unfeasible(x, res):
    """
    Purge unfeasible event locations by setting their interpolation weights to zero.
    :param x: location of motion compensated events
    :param res: resolution of the image space
    :return masked indices
    :return mask for interpolation weights
    """

    mask = torch.ones((x.shape[0], x.shape[1], 1)).to(x.device)
    mask_y = (x[:, :, 1:2] < 0) + (x[:, :, 1:2] >= res[0])
    mask_x = (x[:, :, 0:1] < 0) + (x[:, :, 0:1] >= res[1])
    mask[mask_y + mask_x] = 0
    return x * mask, mask


def get_interpolation(events, flow, tref, res, flow_scaling, round_idx=False):
    '''
    :param events: batch_size * N * 4
    :param flow: batch_size * N * 2
    :param tref: float, warp the events to timestamp tref
    :param res: resolution of the image (H, W)
    :param flow_scaling: scale 1.0
    :param round_idx: whether to interpolate
    :return:
    '''
    # event propagation
    # device = events.device
    #delta = (tref - events[:, :, 3:]) * flow
    warped_events = events[:, :, :2] + (tref - events[:, :, 3:]) * flow * flow_scaling

    if round_idx:
        # no bilinear interpolation
        idx = torch.round(warped_events)
        weights = torch.ones(idx.shape).type_as(events)#.to(device)
    else:
        # get scattering indices
        top_y = torch.floor(warped_events[:, :, 1:2])
        bot_y = torch.floor(warped_events[:, :, 1:2] + 1)
        left_x = torch.floor(warped_events[:, :, 0:1])
        right_x = torch.floor(warped_events[:, :, 0:1] + 1)

        top_left = torch.cat([left_x, top_y], dim=2)
        top_right = torch.cat([right_x, top_y], dim=2)
        bottom_left = torch.cat([left_x, bot_y], dim=2)
        bottom_right = torch.cat([right_x, bot_y], dim=2)
        idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # get scattering interpolation weights
        warped_events = torch.cat([warped_events for i in range(4)], dim=1)
        zeros = torch.zeros(warped_events.shape).type_as(events)#.to(device)
        weights = torch.max(zeros, 1 - torch.abs(warped_events - idx))

    # purge unfeasible indices
    idx, mask = purge_unfeasible(idx, res)

    # make unfeasible weights zero
    weights = torch.prod(weights, dim=-1, keepdim=True) * mask  # bilinear interpolation

    # prepare indices
    idx[:, :, 1] *= res[1]  # torch.view is row-major
    idx = torch.sum(idx, dim=2, keepdim=True)

    return idx, weights


def interpolate(idx, weights, res, polarity_mask=None):
    '''
    interpolate the iwe with weight
    :param idx: batch_size * N the idx of the xy plane
    :param weights: batch_size * weights
    :param res: (H, W)
    :param polarity_mask: polarity mask
    :return: iwe (batch_size, H, W, 1)
    '''
    if polarity_mask is not None:
        weights = weights * polarity_mask
    iwe = torch.zeros((idx.shape[0], res[0] * res[1], 1)).type_as(weights) #.to(weights.device)
    iwe = iwe.scatter_add_(1, idx.long(), weights)
    iwe = iwe.view((idx.shape[0], res[0], res[1], 1))
    return iwe
