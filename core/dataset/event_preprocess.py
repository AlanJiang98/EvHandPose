import numpy as np
import cv2
import imageio
import torch
from tools.basic_io.aedat_preprocess import extract_data_from_aedat4


def undistortion_points(xy, K_old, dist, K_new=None, set_bound=False, width=346, height=240):
    '''
    :param xy: N*2 array of event coordinates
    :param K_old: camera intrinsics
    :param dist: distortion coefficients
        such as
        mtx = np.array(
            [[252.91294004, 0, 129.63181808],
            [0, 253.08270535, 89.72598511],
            [0, 0, 1.]])
        dist = np.array(
            [-3.30783118e+01,  3.40196626e+02, -3.19491618e-04, -6.28058571e-04,
            1.67319020e+02, -3.27436981e+01,  3.29048638e+02,  2.85123812e+02,
            0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00])
    :param K_new: new K for camera intrinsics
    :param set_bound: if true, set the undistorted points bounds
    :return: undistorted points
    '''
    # this function only outputs the normalized point coordinated, so we need apply a projection matrix K
    assert (xy.shape[1] == 2)
    xy = xy.astype(np.float32)
    if K_new is None:
        K_new = K_old
    und = cv2.undistortPoints(src=xy, cameraMatrix=K_old, distCoeffs=dist, P=K_new)
    und = und.reshape(-1, 2)
    und = und[:, :2]
    legal_indices = (und[:, 0] >= 0) * (und[:, 0] <= width-1) * (und[:, 1] >= 0) * (und[:, 1] <= width-1)
    if set_bound:
        und[:, 0] = np.clip(und[:, 0], 0, width-1)
        und[:, 1] = np.clip(und[:, 1], 0, height-1)
    return und, legal_indices


def remove_unfeasible_events(events, height, width):
    x_mask = (events[:, 0] >=0) * (events[:, 0] <= width-1)
    y_mask = (events[:, 1] >=0) * (events[:, 1] <= height-1)
    mask = x_mask * y_mask
    return mask

def event_count_to_frame(xy, weight, height=260, width=346, interpolate=False):
    img = torch.zeros((height, width), dtype=torch.float32).to(xy.device)
    if interpolate:
        xys, weights = get_intepolate_weight(xy, weight, height, width)
    else:
        xys, weights = xy, weight
    if xys.dtype is not torch.long:
        xys = xys.clone().long()
    img.index_put_((xys[:, 1], xys[:, 0]), weights, accumulate=True)
    return img


def event_to_channels(event_tmp, height=260, width=346, is_neg=False, interpolate=False):
    mask_pos = event_tmp[:, 2] == 1
    mask_neg = event_tmp[:, 2] == 0
    pos_img = event_count_to_frame(event_tmp[:, :2], (1*mask_pos).float(), height, width, interpolate)
    neg_img = event_count_to_frame(event_tmp[:, :2], (1*mask_neg).float(), height, width, interpolate)
    if is_neg:
        neg_img *= -1
    return torch.stack([pos_img, neg_img])


def event_to_voxel(event_tmp, num_bin, height=260, width=346, interpolate=False):
    voxel = []
    ps = event_tmp[:, 2] * 2 - 1
    ts = event_tmp[:, 3]
    ts = ts * (num_bin - 1)
    zeros = torch.zeros(ts.shape, dtype=torch.float32)
    for bin_id in range(num_bin):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - bin_id))
        voxel_tmp = event_count_to_frame(event_tmp[:, :2], weights * ps, height, width, interpolate)
        voxel.append(voxel_tmp)
    return torch.stack(voxel)


def get_intepolate_weight(events, weight, height, width, is_LNES=False):
    top_y = torch.floor(events[:, 1:2])
    bot_y = torch.floor(events[:, 1:2] + 1)
    left_x = torch.floor(events[:, 0:1])
    right_x = torch.floor(events[:, 0:1] + 1)

    top_left = torch.cat([left_x, top_y], dim=1)
    top_right = torch.cat([right_x, top_y], dim=1)
    bottom_left = torch.cat([left_x, bot_y], dim=1)
    bottom_right = torch.cat([right_x, bot_y], dim=1)

    idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=0)
    events_tmp = torch.cat([events for i in range(4)], dim=0)
    zeros = torch.zeros(idx.shape).type_as(idx)
    weights_bi = torch.max(zeros, 1 - torch.abs(events_tmp[:, :2] - idx))
    mask = remove_unfeasible_events(idx, height, width)
    weight_ori = torch.cat([weight for i in range(4)], dim=0)
    events_tmp[:, :2] = idx
    if is_LNES:
        weights_bi_tmp = torch.prod(weights_bi, dim=-1)
        mask *= (weights_bi_tmp != 0)
        weights_lnes = events_tmp[:, 3][mask]
        events_tmp = events_tmp[mask]
        weights_final, indices = torch.sort(weights_lnes, dim=0, descending=False)
        events_final = events_tmp[indices]
        return events_final, weights_final
    else:
        weights_final = torch.prod(weights_bi, dim=-1) * mask * weight_ori
        return events_tmp[mask], weights_final[mask]


def event_to_LNES(event_tmp, height=260, width=346, interpolate=False):
    ts = event_tmp[:, 3]
    img = torch.zeros((2, height, width), dtype=torch.float32)
    if interpolate:
        events, weights = get_intepolate_weight(event_tmp, ts, height, width, is_LNES=True)
    else:
        events, weights = event_tmp, ts
    if events.dtype is not torch.long:
        xyp = events[:, :3].clone().long()
    img[xyp[:, 2], xyp[:, 1], xyp[:, 0]] = weights
    return img[[1, 0], :, :]

def create_polarity_mask(polarity):
    pos = polarity == 1
    neg = polarity == 0
    event_pol_mask = torch.stack([pos, neg])
    return event_pol_mask.transpose(1, 0)

