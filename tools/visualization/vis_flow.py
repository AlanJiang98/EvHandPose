import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def events_to_image(inp_events, color_scheme="green_red", background='black'):
    """
    Visualize the input events.
    :param inp_events: [batch_size x H x W x 2] per-pixel and per-polarity event count
    :param color_scheme: green_red/gray
    :return event_image: [H x W x 3] color-coded event image
    """

    # if inp_events.max() > 1.5:
    #     inp_events = inp_events / inp_events.max()
    inp_events = np.clip(inp_events, 0, 1)

    if color_scheme == "gray":
        event_image = inp_events[..., 0] * 0.5 + inp_events[..., 1] * 0.5
    elif color_scheme == "green_red":
        zero_shape = np.array(inp_events.shape)
        zero_shape[-1] = 1
        event_image = np.concatenate([inp_events, np.zeros(zero_shape)], axis=-1)
    if background == 'white':
        blank = np.prod(event_image == 0, axis=-1)
        event_image[blank.astype(np.bool)] = 1
    return event_image


def flow_to_image(flow_x, flow_y):
    """
    Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
    Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
    Paredes-Valles et al., CVPR'21.
    :param flow_x: [H x W x 1] horizontal optical flow component
    :param flow_y: [H x W x 1] vertical optical flow component
    :return flow_rgb: [H x W x 3] color-encoded optical flow
    """
    flows = np.stack((flow_x, flow_y), axis=2)
    mag = np.linalg.norm(flows, axis=2)
    min_mag = np.min(mag)
    mag_range = np.max(mag) - min_mag

    ang = np.arctan2(flow_y, flow_x) + np.pi
    ang *= 1.0 / np.pi / 2.0

    hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
    hsv[:, :, 0] = ang
    hsv[:, :, 1] = 1.0
    hsv[:, :, 2] = mag - min_mag
    if mag_range != 0.0:
        hsv[:, :, 2] /= mag_range

    flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
    # blank = np.prod(flow_rgb == 0, axis=-1)
    # flow_rgb[blank.astype(np.bool)] = 1
    return flow_rgb


def vis_event_frames(data_: torch.tensor, save_dir:str, filename_list=None, video_name=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = data_.shape[0]
    data = data_.clone()
    frames_np = data.detach().cpu().numpy()
    if filename_list is not None:
        assert len(filename_list) == batch_size
    else:
        filename_list = [str(i)+'.jpg' for i in range(batch_size)]
    file_paths = [os.path.join(save_dir, filename) for filename in filename_list]
    frame_list = []
    for i in range(batch_size):
        frame = events_to_image(frames_np[i], color_scheme='green_red')
        frame_list.append((frame * 255).astype(np.uint8))
    for i in range(batch_size):
    #     plt.imshow(frame_list[i])
    #     plt.show()
        imageio.imwrite(file_paths[i], frame_list[i])
    frame_stacked = np.vstack([frame[None, ...] for frame in frame_list])
    video_name = 'output.mp4' if video_name is None else video_name
    imageio.mimwrite(os.path.join(save_dir, video_name), frame_stacked, fps=15, quality=8, macro_block_size=1)


def vis_rendered_frames(data_: torch.tensor, save_dir:str, filename_list=None, video_name=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = data_.shape[0]
    data = data_.clone()
    frames_np = data.detach().cpu().numpy()
    if filename_list is not None:
        assert len(filename_list) == batch_size
    else:
        filename_list = [str(i)+'.jpg' for i in range(batch_size)]
    file_paths = [os.path.join(save_dir, filename) for filename in filename_list]
    frame_list = []
    for i in range(batch_size):
        frame_list.append((np.clip(frames_np[i] * 255, 0, 255)).astype(np.uint8))
    for i in range(batch_size):
    #     plt.imshow(frame_list[i])
    #     plt.show()
        imageio.imwrite(file_paths[i], frame_list[i])
    frame_stacked = np.vstack([frame[None, ...] for frame in frame_list])
    video_name = 'output.mp4' if video_name is None else video_name
    imageio.mimwrite(os.path.join(save_dir, video_name), frame_stacked, fps=15, quality=8, macro_block_size=1)

def vis_kps_2d(kps_2d, frames, save_dir, video_name=None):
    HAND_JOINT_COLOR_Lst = [(255, 255, 255),
                            (255, 192, 203), (255, 192, 203), (255, 192, 203), (255, 192, 203),
                            (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
                            (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
                            (128, 0, 128), (128, 0, 128), (128, 0, 128), (128, 0, 128),
                            (128, 128, 0), (128, 128, 0), (128, 128, 0), (128, 128, 0)]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = kps_2d.shape[0]
    kps_2d_np = kps_2d.detach().cpu().numpy()
    if type(frames) == torch.Tensor:
        frames_np = frames.detach().cpu().numpy()
    else:
        frames_np = frames
    filename_list = [str(i)+'.jpg' for i in range(batch_size)]
    file_paths = [os.path.join(save_dir, filename) for filename in filename_list]
    frame_list = []
    for i in range(batch_size):
        img = np.ascontiguousarray(frames_np[i] * 255, dtype=np.uint8)
        for joint_idx, kp in enumerate(kps_2d_np[i]):
            cv2.circle(img, (int(kp[0]), int(kp[1])), 2, HAND_JOINT_COLOR_Lst[joint_idx], -1)
            if joint_idx % 4 == 1:
                cv2.line(img, (int(kps_2d_np[i][0][0]), int(kps_2d_np[i][0][1])), (int(kp[0]), int(kp[1])),
                         HAND_JOINT_COLOR_Lst[joint_idx], 1)  # connect to root
            elif joint_idx != 0:
                cv2.line(img, (int(kps_2d_np[i][joint_idx - 1][0]), int(kps_2d_np[i][joint_idx - 1][1])), (int(kp[0]), int(kp[1])),
                         HAND_JOINT_COLOR_Lst[joint_idx], 1)  # connect to root
        frame_list.append(img)
    for i in range(batch_size):
        imageio.imwrite(file_paths[i], frame_list[i])
    frame_stacked = np.vstack([frame[None, ...] for frame in frame_list])
    video_name = 'output.mp4' if video_name is None else video_name
    imageio.mimwrite(os.path.join(save_dir, video_name), frame_stacked, fps=15, quality=8, macro_block_size=1)

def vis_edges(image_lens: torch.tensor, edges: torch.tensor ,save_dir:str, filename_list=None, video_name=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    batch_size = image_lens.shape[0]
    frames_np = image_lens.detach().cpu().numpy()
    edges = edges.detach().cpu().numpy()
    if filename_list is not None:
        assert len(filename_list) == batch_size
    else:
        filename_list = [str(i)+'.jpg' for i in range(batch_size)]
    file_paths = [os.path.join(save_dir, filename) for filename in filename_list]
    frame_list = []
    for i in range(batch_size):
        frame = events_to_image(frames_np[i], color_scheme='green_red')
        edges_tmp = (edges[i] / (edges[i].max()+1e-6)).repeat(3, axis=2)
        frame += edges_tmp * 3
        frame = np.clip(frame, 0, 1)
        frame_list.append((frame * 255).astype(np.uint8))
    for i in range(batch_size):
    #     plt.imshow(frame_list[i])
    #     plt.show()
        imageio.imwrite(file_paths[i], frame_list[i])
    frame_stacked = np.vstack([frame[None, ...] for frame in frame_list])
    video_name = 'output.mp4' if video_name is None else video_name
    imageio.mimwrite(os.path.join(save_dir, video_name), frame_stacked, fps=15, quality=8, macro_block_size=1)

def vis_flow_frames(data_:torch.tensor, save_dir:str, filename_list=None, video_name=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = data_.clone()
    batch_size = data.shape[0]
    frames_np = data.detach().cpu().numpy()
    if filename_list is not None:
        assert len(filename_list) == batch_size
    else:
        filename_list = [str(i)+'.jpg' for i in range(batch_size)]
    file_paths = [os.path.join(save_dir, filename) for filename in filename_list]
    frame_list = []
    for i in range(batch_size):
        # frame = flow_to_color(frames_np[i])
        # frame_list.append(frame.astype(np.uint8))
        frame = flow_to_image(frames_np[i, :, :, 0], frames_np[i, :, :, 1])
        frame_list.append((frame*255).astype(np.uint8))
    for i in range(batch_size):
        # plt.imshow(frame_list[i])
        # plt.show()
        imageio.imwrite(file_paths[i], frame_list[i])
    frame_stacked = np.vstack([frame[None, ...] for frame in frame_list])
    video_name = 'output_flow.mp4' if video_name is None else video_name
    imageio.mimwrite(os.path.join(save_dir, video_name), frame_stacked, fps=15, quality=8, macro_block_size=1)

def draw_arrow_per_pixel(image, x, y, optical_flow):
    height, width, _ = image.shape
    start_point = np.array([x, y])
    if np.sum(optical_flow**2) < 1:
        return
    end_point = start_point + optical_flow
    end_point = np.round(end_point).astype(np.int32)
    end_point = np.clip(end_point, 0, [width - 1, height - 1])
    assert end_point[0] < width
    assert end_point[1] < height
    cv2.arrowedLine(image, tuple(start_point), tuple(end_point), (1.0, 1.0, 1.0), thickness=1)

def draw_arrow_image(image, optical_flow, step=6):
    height, width, _ = image.shape
    assert optical_flow.shape[0] == height
    assert optical_flow.shape[1] == width
    assert optical_flow.shape[2] == 2
    assert image.max() <= 1
    for y in np.arange(0, height, step):
        assert y < height
        for x in np.arange(0, width, step):
            assert x < width
            draw_arrow_per_pixel(image, x, y, optical_flow[y, x, :])
    return image


def vis_flow_arrow(event_frames_, flow_, save_dir:str, filename_list=None, video_name=None, step=4):
    '''
    draw flow for better visualization
    :return:
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    event_frames = event_frames_.clone()
    flow = flow_.clone()
    batch_size = event_frames.shape[0]
    events_np = event_frames.detach().cpu().numpy()
    flows_np = flow.detach().cpu().numpy()
    if filename_list is not None:
        assert len(filename_list) == batch_size
    else:
        filename_list = [str(i)+'.jpg' for i in range(batch_size)]
    file_paths = [os.path.join(save_dir, filename) for filename in filename_list]
    frame_list = []
    for i in range(batch_size):
        image = np.ascontiguousarray(events_to_image(events_np[i]))
        frame = draw_arrow_image(image, flows_np[i], step=step)
        frame_list.append((frame*255).astype(np.uint8))
    for i in range(batch_size):
        imageio.imwrite(file_paths[i], frame_list[i])
    frame_stacked = np.vstack([frame[None, ...] for frame in frame_list])
    video_name = 'output_flow_arrow.mp4' if video_name is None else video_name
    imageio.mimwrite(os.path.join(save_dir, video_name), frame_stacked, fps=15, quality=8, macro_block_size=1)

