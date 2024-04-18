import os
from abc import abstractmethod
import copy
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import Dataset
from tools.basic_io.json_utils import json_read
from core.dataset.noise_filter import background_activity_filter
from tools.basic_io.aedat_preprocess import extract_data_from_aedat4
from core.dataset.joint_indices import indices_change
from core.dataset.event_preprocess import undistortion_points, event_to_voxel, event_to_LNES, event_to_channels, create_polarity_mask, remove_unfeasible_events
from tools.visualization.vis_flow import *
import roma
from scipy.interpolate import interp1d

class BaseDataset(Dataset):
    def __init__(self, config):
        # index of the dataset
        self.index = 0
        # config of the dataset
        self.config = config
        # length of the sequences
        self.len = 0
        # original annotation dicts
        self.annot = None
        # mode
        self.mode = 'train'
        # event array
        self.events = None
        # 3D GT frame id and timestamps
        self.annot_info = []
        # right or left
        self.hand_type = 'right'
        # normal, highlight, flash
        self.scene = 'normal'
        # normal, fast
        self.motion_type = 'normal'
        # time synchronization
        # annot_time + self.time_sync = event_time
        self.time_sync = 0
        # used to check the experiment type
        self.exper = None
        # used to split the evaluation data
        self.annoted = False
        # all the seq data
        self.seq_items = []
        # interpolation function for bbox generation
        self.bbox_inter_f = None
        self.bbox_seq = None
        # time window for event sequence
        self.time_window = 0
        self.left_flip = False
        self.mano_key = 'mano'

    def set_index(self, index):
        self.index = index

    def regenerate_items(self):
        if self.mode != 'eval':
            self.seq_items = []
        self.generate_items()

    @abstractmethod
    def load_annotations(self):
        '''
        load annotations of sequence
        if this dataset is for semisupervised, self.annoted=Flase
        :return: self.annot, self.annoted
        '''
        pass

    @abstractmethod
    def load_events(self):
        '''
        load events as N*4 array and filter the noise if activity_filter is True
        timestamps are in milliseconds
        :return: self.events
        '''
        pass

    @abstractmethod
    def get_annotations(self, id:str):
        '''
        get annotations of mano by the image id
        :param id: image id
        :return: annotations dict
        '''
        pass

    @abstractmethod
    def process_annotations(self):
        '''
        get mano id and do time synchronization
        :return: self.annot_info
        '''
        pass

    def add_noise(self, data, mean=0., sigma=0.2, salt_pepper=0.02):
        """
        get noise augmentation
        :param data: torch.tensor
        :param mean: mean of gaussian noise
        :param sigma: variance of gaussian noise
        :param salt_pepper: the probability of salt_pepper noise
        :return:
        """
        shape_ = data.shape
        gaussian_noise = torch.randn(*shape_) * sigma + mean
        salt_pepper_noise = torch.rand(*shape_) < salt_pepper
        return data + gaussian_noise + salt_pepper_noise

    def get_interpolate_joints_3d(self, t0):
        '''
        get interpolated 3d joints at time t0
        :param t0:
        :return: joints
        '''
        if t0 < self.joints_3d_bbox[0][0]:
            t0 = self.joints_3d_bbox[0][0]
        elif t0 > self.joints_3d_bbox[0][-1]:
            t0 = self.joints_3d_bbox[0][-1]
        joints = self.bbox_inter_f(t0)
        return joints

    def get_bbox_from_joints_3d(self, K, joints):
        # get bbox from 3D joints
        kps = np.dot(K, joints.transpose(1, 0)).transpose(1, 0)
        kps = kps[:, :2] / kps[:, 2:]
        x_min = np.min(kps[:, 0])
        x_max = np.max(kps[:, 0])
        y_min = np.min(kps[:, 1])
        y_max = np.max(kps[:, 1])
        return np.array([x_min, x_max, y_min, y_max], dtype=np.float32)

    def get_interpolate_bbox_2d(self, t0):
        # get bbox at t0 for fast sequences
        if t0 < self.bbox_seq[0, 0]:
            t0 = self.bbox_seq[0, 0]
        elif t0 > self.bbox_seq[-1, 0]:
            t0 = self.bbox_seq[-1, 0]
        bbox = self.bbox_inter_f(t0)
        center = bbox[:2]
        size = bbox[2]
        return center, size

    def get_bbox_from_time_slice(self, t0, t1, K):
        # if fast motion, use bbox sequences interpolation
        # if not, use 3D joints for bbox generation
        # we want the bbox to cover the events from t0 to t1,
        # so the bbox is max(bbox_t0, bbox_t1)
        if self.annoted:
            joints_0 = self.get_interpolate_joints_3d(t0)
            joints_1 = self.get_interpolate_joints_3d(t1)
            K_ = np.array(K)
            bbox1 = self.get_bbox_from_joints_3d(K_, joints_0)
            bbox2 = self.get_bbox_from_joints_3d(K_, joints_1)
            x_min = np.min([bbox1[0], bbox2[0]])
            x_max = np.max([bbox1[1], bbox2[1]])
            y_min = np.min([bbox1[2], bbox2[2]])
            y_max = np.max([bbox1[3], bbox2[3]])
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            ori_size = np.max([x_max-x_min, y_max-y_min])
            bbox_size = self.config['preprocess']['bbox']['joints_to_bbox_rate'] * ori_size
        else:
            center0, size0 = self.get_interpolate_bbox_2d(t0)
            center1, size1 = self.get_interpolate_bbox_2d(t1)
            top_left_0 = center0 - size0 / 2
            bottom_right_0 = center0 + size0 / 2
            top_left_1 = center1 - size1 / 2
            bottom_right_1 = center1 + size1 / 2
            top_left = np.min([top_left_0, top_left_1], axis=0)
            bottom_right = np.max([bottom_right_0, bottom_right_1], axis=0)
            center_x = (top_left[0] + bottom_right[0]) / 2.0
            center_y = (top_left[1] + bottom_right[1]) / 2.0
            bbox_size = np.max(bottom_right - top_left) # * self.config['preprocess']['bbox']['joints_to_bbox_rate']
        return torch.tensor([bbox_size, center_x, center_y], dtype=torch.float32)

    def generate_items(self):

        flow_flag = self.config['method']['flow']['train'] and (self.mode != 'eval' or (self.mode == 'eval' and self.config['log']['gt_show'] and self.motion_type != 'fast'))
        if self.mode == 'train' or flow_flag:
            if 'supervision' in self.config['exper']['supervision'] or self.config['method']['flow']['train']:
                if self.annoted:
                    # get the supervision sequences
                    # seq_len: sequence length for each model; annot_len: number of annotations each sequence
                    for i in range(len(self.annot_info) - self.config['method']['seq_model']['annot_len']):
                        seq_item = []
                        annot_tmp = self.annot_info[i:i+self.config['method']['seq_model']['annot_len']]
                        time_interval = np.zeros(self.config['method']['seq_model']['annot_len'] - 1)
                        for j in range(self.config['method']['seq_model']['annot_len']-1):
                            time_interval[j] = int(annot_tmp[j+1]['image_id']) - int(annot_tmp[j]['image_id'])
                            # to check whether the annotated data are existed continuously
                        if (time_interval == time_interval[0]).all():
                            # generated supervised data in the same time window slice
                            start_times = np.arange(-self.config['method']['seq_model']['seq_len'], 0) * 1000000. / self.config['data']['fps']\
                                          + annot_tmp[-1]['timestamp']
                            end_times = start_times + 1000000. / self.config['data']['fps']
                            start_indices = np.searchsorted(self.events[:, 3], start_times)
                            if start_indices[0] == 0:
                                continue
                            end_indices = np.searchsorted(self.events[:, 3], end_times)
                            if (end_indices - start_indices < 100).any():
                                continue
                            marker = self.config['method']['seq_model']['seq_len'] - self.config['method']['seq_model']['annot_len']
                            for k in range(self.config['method']['seq_model']['seq_len']):
                                item = {}
                                item['start_timestamp'] = start_times[k]
                                item['end_timestamp'] = end_times[k]
                                item['start_index'] = start_indices[k]
                                item['end_index'] = end_indices[k]
                                if k < marker:
                                    item['id'] = '-1'
                                else:
                                    item['id'] = annot_tmp[k - marker]['image_id']
                                seq_item.append(item)
                            self.seq_items.append(seq_item)

            if 'semi-supervision' in self.config['exper']['supervision']:
                if self.annoted:
                    if self.annot_info[-5]['timestamp'] > self.config['method']['seq_model']['seq_len'] * self.config['preprocess']['max_window_time']:
                        N = int(len(self.annot_info) * self.config['preprocess']['annot_use_rate'])
                        count = 0
                        while count < N:
                            idx = np.random.randint(len(self.annot_info))
                            seq_item = []
                            # given an annotated timestamp,
                            # generated a sequence of window timestamp markers for sequence data
                            step = self.time_window + np.random.randn() * self.config['preprocess']['window_time_var']
                            step = np.clip(step, self.config['preprocess']['min_window_time'], self.config['preprocess']['max_window_time'])
                            timestamps = np.arange(-self.config['method']['seq_model']['seq_len']+1, 2) * step + self.annot_info[idx]['timestamp']
                            indices = np.searchsorted(self.events[:, 3], timestamps)
                            if (indices[1:] - indices[:-1] < 150).any():
                                continue
                            if indices[0] == 0 or indices[-1] >= self.events.shape[0] - 1:
                                continue
                            for k in range(self.config['method']['seq_model']['seq_len']):
                                item = {}
                                item['start_timestamp'] = timestamps[k]
                                item['end_timestamp'] = timestamps[k+1]
                                item['start_index'] = indices[k]
                                item['end_index'] = indices[k+1]
                                tmp_len = self.config['method']['seq_model']['annot_len'] // 2 + 1
                                if k == self.config['method']['seq_model']['seq_len'] - tmp_len:
                                    item['id'] = self.annot_info[idx]['image_id']
                                else:
                                    item['id'] = '-1'
                                seq_item.append(item)
                            self.seq_items.append(seq_item)
                            count += 1
            self.len = len(self.seq_items)
        elif self.mode == 'val' or self.mode == 'eval':
            # for validation, we only need the last item annotation of a sequence
            valid_annot = []
            for i in range(len(self.annot_info)):
                if self.annot_info[i]['timestamp'] - self.config['method']['seq_model']['seq_len'] * \
                        self.config['preprocess']['max_window_time'] > self.events[0, 3] and \
                        self.annot_info[i]['timestamp'] < self.events[-1, 3]:
                    valid_annot.append(i)
            self.len = len(valid_annot)
            for i in valid_annot:
                seq_item = []
                annot_tmp = self.annot_info[i]
                # generated supervised data in the same time window slice,
                if self.motion_type == 'fast':
                    step = self.config['preprocess']['same_time_window_fast']
                else:
                    step = self.time_window
                timestamps = np.arange(-self.config['method']['seq_model']['seq_len'], 1) * step\
                              + annot_tmp['timestamp']
                indices = np.searchsorted(self.events[:, 3], timestamps)
                if indices[0] == 0:
                    raise ValueError(
                        'Invalid timestamp at dataset {} at annot id {} timestamp {}'.format(self.index, i,
                                                                                             annot_tmp[
                                                                                                 'timestamp']))
                for k in range(self.config['method']['seq_model']['seq_len']):
                    item = {}
                    item['start_timestamp'] = timestamps[k]
                    item['end_timestamp'] = timestamps[k+1]
                    item['start_index'] = indices[k]
                    item['end_index'] = indices[k+1]
                    if self.motion_type == 'fast':
                        if k != self.config['method']['seq_model']['seq_len']-1:
                            item['id'] = '-1'
                        else:
                            item['id'] = annot_tmp['image_id']
                    else:
                        if k < self.config['method']['seq_model']['seq_len']-2:
                            item['id'] = '-1'
                        else:
                            item['id'] = self.annot_info[i-(self.config['method']['seq_model']['seq_len']-1-k)]['image_id']
                    seq_item.append(item)
                self.seq_items.append(seq_item)
            self.len = len(self.seq_items)

    def __getitem__(self, idx):
        # avoid the idx out of range
        idx = idx % len(self.seq_items)
        seq_item = self.seq_items[idx]
        seq_len = len(seq_item)
        output_tmp = {}
        output_tmp['start_timestamps'] = torch.zeros(seq_len, dtype=torch.float32)
        output_tmp['end_timestamps'] = torch.zeros(seq_len, dtype=torch.float32)
        output_tmp['start_indices'] = torch.zeros(seq_len, dtype=torch.long)
        output_tmp['end_indices'] = torch.zeros(seq_len, dtype=torch.long)
        # number of events in each interval
        output_tmp['Num_events'] = torch.zeros(seq_len, dtype=torch.int32)
        output_tmp['ids'] = torch.zeros(seq_len, dtype=torch.int32)
        output_tmp['dataset_index'] = torch.tensor(self.index, dtype=torch.int32)
        output_tmp['events'] = torch.zeros((seq_len, self.config['preprocess']['num_events'], 4), dtype=torch.float32)

        # polarity mask
        output_tmp['pol_mask'] = torch.zeros((seq_len, self.config['preprocess']['num_events'], 2), dtype=torch.float32)
        output_tmp['annot'] = {}
        output_tmp['annot']['mano_rot_pose'] = torch.zeros((seq_len, 3), dtype=torch.float32)
        output_tmp['annot']['mano_hand_pose'] = torch.zeros((seq_len, 45), dtype=torch.float32)
        output_tmp['annot']['mano_shape'] = torch.zeros((seq_len, 10), dtype=torch.float32)
        output_tmp['annot']['mano_trans'] = torch.zeros((seq_len, 3), dtype=torch.float32)
        output_tmp['annot']['joint'] = torch.zeros((seq_len, 21, 3), dtype=torch.float32)
        output_tmp['annot']['joint_valid'] = torch.zeros((seq_len, 21, 1), dtype=torch.float32)
        output_tmp['annot']['hand_type'] = torch.zeros(seq_len, dtype=torch.int32)
        output_tmp['annot']['index'] = torch.zeros(seq_len, dtype=torch.int32)
        output_tmp['annot']['fast'] = torch.zeros(seq_len, dtype=torch.int32)
        output_tmp['annot']['t'] = torch.zeros((seq_len, 3), dtype=torch.float32)
        output_tmp['annot']['K'] = torch.zeros((seq_len, 3, 3), dtype=torch.float32)
        output_tmp['annot']['R'] = torch.zeros((seq_len, 3, 3), dtype=torch.float32)
        output_tmp['annot']['2d_joints'] = torch.zeros((seq_len, 21, 2), dtype=torch.float32)

        output_tmp['bbox'] = torch.zeros((seq_len, 3), dtype=torch.float32)
        output_tmp['eci'] = torch.zeros((seq_len,
                                        2,
                                        self.config['data']['height'],
                                        self.config['data']['width']), dtype=torch.float32)
        output_tmp['origin_eci'] = torch.zeros((seq_len,
                                         2,
                                         self.config['data']['height'],
                                         self.config['data']['width']), dtype=torch.float32)
        output_tmp['eci_with_aug'] = torch.zeros((seq_len,
                                         2,
                                         self.config['data']['height'],
                                         self.config['data']['width']), dtype=torch.float32)
        # check whether to use bbox
        if self.config['preprocess']['bbox']['usage']:
            input_size = (self.config['preprocess']['bbox']['size'], self.config['preprocess']['bbox']['size'])
        else:
            input_size = (self.config['data']['height'], self.config['data']['width'])

        tmp_len = self.config['method']['event_encoder']['num_bins'] * 2
        output_tmp['event_encoder_repre'] = torch.zeros((seq_len,
                                                        tmp_len,
                                                        input_size[0],
                                                         input_size[1]), dtype=torch.float32)

        if self.config['method']['flow']['usage']:
            output_tmp['flow_repre'] = torch.zeros((seq_len,
                                                    self.config['method']['flow']['num_bins'],
                                                    input_size[0],
                                                    input_size[1]), dtype=torch.float32)

        output_tmp['crop_eci'] = torch.zeros((seq_len,
                                                    2,
                                                    input_size[0],
                                                    input_size[1]), dtype=torch.float32)

        # for training, generate the augmentation parameters
        # make sure for a sequence, the parameters keep the same
        if self.mode == 'train' and seq_item[-1]['id'] != '-1':# and self.config['data']['dataset'] != 'EventHand':
            annot_tmp = self.get_annotations(seq_item[-1]['id'])
            trans_new = annot_tmp['mano_trans'][2] + (torch.rand(1) - 0.5) * 2 * self.config['preprocess']['augment']['scale_var']
            trans_new = torch.clip(trans_new, self.config['preprocess']['augment']['depth_range'][0], self.config['preprocess']['augment']['depth_range'][1])
            # scale, trans, rotation augmentation
            scale = annot_tmp['mano_trans'][2] / trans_new
            c_xy = annot_tmp['K'][:2, 2]
            theta = (torch.rand(1) - 0.5) * 2. * self.config['preprocess']['augment']['rotate_var']
            trans_xy = (torch.rand(2) - 0.5) * 2. * self.config['preprocess']['augment']['trans_var']
            trans_xy[0] = torch.clip(trans_xy[0],
                                     self.config['preprocess']['augment']['trans_range_x'][0] - annot_tmp['mano_trans'][0],
                                     self.config['preprocess']['augment']['trans_range_x'][1] - annot_tmp['mano_trans'][0])
            trans_xy[1] = torch.clip(trans_xy[1],
                                     self.config['preprocess']['augment']['trans_range_y'][0] - annot_tmp['mano_trans'][1],
                                     self.config['preprocess']['augment']['trans_range_y'][1] - annot_tmp['mano_trans'][1])

        # generate data for each interval
        for i, item in enumerate(seq_item):
            output_tmp['start_timestamps'][i] = item['start_timestamp']
            output_tmp['end_timestamps'][i] = item['end_timestamp']
            output_tmp['start_indices'][i] = item['start_index']
            output_tmp['end_indices'][i] = item['end_index']
            output_tmp['ids'][i] = int(item['id'])

            annot_tmp = self.get_annotations(item['id'])
            # assign annotations
            for key in annot_tmp.keys():
                if key != 'root_joint':
                    output_tmp['annot'][key][i] = annot_tmp[key]

            tmp_events = self.events[item['start_index']:item['end_index'], :].copy()
            # normalize the time to [0, 1]
            tmp_events[:, 3] = (tmp_events[:, 3] - item['start_timestamp']) / (item['end_timestamp'] - item['start_timestamp'])
            tmp_events = torch.tensor(tmp_events, dtype=torch.float32)

            # this is the eci without augmentation
            output_tmp['origin_eci'][i] = event_to_channels(tmp_events, height=self.config['data']['height'], width=self.config['data']['width'])

            # get bbox
            bbox = self.get_bbox_from_time_slice(item['start_timestamp'], item['end_timestamp'], annot_tmp['K'])

            # augmentations
            # for annotated data when training
            if self.mode == 'train' and seq_item[-1]['id'] != '-1':# and self.config['data']['dataset'] != 'EventHand':
                # scale augmentation
                tmp_events[:, :2] = (tmp_events[:, :2] - c_xy[None, :]) * scale + c_xy[None, :]

                output_tmp['annot']['2d_joints'][i] = (output_tmp['annot']['2d_joints'][i] - c_xy[None, :]) * scale + c_xy[None, :]
                output_tmp['annot']['mano_trans'][i][2] /= scale[0]
                output_tmp['annot']['joint'][i, :, 2] /= scale[0]

                bbox[1:] = (bbox[1:] - c_xy) * scale + c_xy
                bbox[0] = bbox[0] * scale

                # rotation augmentation
                R_augment = roma.rotvec_to_rotmat(theta*torch.tensor([0.0, 0.0, 1.0]))
                tmp_events[:, :2] = torch.mm(R_augment[:2, :2], (tmp_events[:, :2] - c_xy[None, :]).transpose(0, 1)).transpose(0, 1) + c_xy[None, :]

                output_tmp['annot']['2d_joints'][i] = torch.mm(R_augment[:2, :2],
                                                               (output_tmp['annot']['2d_joints'][i] - c_xy[None, :]).transpose(0, 1)).transpose(0, 1) + c_xy[None, :]
                bbox[1:] = torch.mm(R_augment[:2, :2], (bbox[1:] - c_xy)[..., None])[:, 0] + c_xy
                R_mano_rot = roma.rotvec_to_rotmat(output_tmp['annot']['mano_rot_pose'][i])
                R_mano_rot_new = torch.mm(R_augment, R_mano_rot)
                rot_pose_new = roma.rotmat_to_rotvec(R_mano_rot_new)
                output_tmp['annot']['mano_rot_pose'][i] = rot_pose_new
                trans_new = torch.mm(R_augment, (output_tmp['annot']['mano_trans'][i] + annot_tmp['root_joint'])[:, None])[:, 0] - annot_tmp['root_joint']
                output_tmp['annot']['mano_trans'][i] = trans_new
                output_tmp['annot']['joint'][i, :, :2] = torch.mm(R_augment[:2, :2], output_tmp['annot']['joint'][i, :, :2].transpose(0, 1)).transpose(0, 1)

                # translation
                image_trans_x = output_tmp['annot']['K'][i, 0, 0] * trans_xy[0] / output_tmp['annot']['mano_trans'][i][2]
                image_trans_y = output_tmp['annot']['K'][i, 1, 1] * trans_xy[1] / output_tmp['annot']['mano_trans'][i][2]
                tmp_events[:, :1] += image_trans_x
                tmp_events[:, 1:2] += image_trans_y

                output_tmp['annot']['2d_joints'][i, :, 0] += image_trans_x
                output_tmp['annot']['2d_joints'][i, :, 1] += image_trans_y

                bbox[1] += image_trans_x
                bbox[2] += image_trans_y

                output_tmp['annot']['mano_trans'][i, :2] += trans_xy
                output_tmp['annot']['joint'][i, :, :2] += trans_xy[None, :]

            # remove events out of the image plane
            mask = remove_unfeasible_events(tmp_events, self.config['data']['height'], self.config['data']['width'])
            tmp_events = tmp_events[mask]
            output_tmp['Num_events'][i] = tmp_events.shape[0]

            unbbox_events = tmp_events.clone()
            output_tmp['eci_with_aug'][i] = event_to_channels(unbbox_events, height=self.config['data']['height'], width=self.config['data']['width'])

            if self.config['preprocess']['bbox']['usage']:
                # scale and trans for bbox
                c_xy = annot_tmp['K'][:2, 2]
                if self.config['preprocess']['bbox']['resize']:
                    bbox_scale = self.config['preprocess']['bbox']['size'] / bbox[0]
                else:
                    bbox_scale = 1.0
                center_xy = bbox[1:]
                center_xy = (center_xy - c_xy) * bbox_scale + c_xy
                top_left = center_xy - self.config['preprocess']['bbox']['size'] / 2
                if self.mode == 'train':
                    top_left += (torch.rand(2)-0.5) * 2 * self.config['preprocess']['bbox']['size'] * self.config['preprocess']['bbox']['shift_var']
                output_tmp['bbox'][i, 0] = bbox_scale
                output_tmp['bbox'][i, 1:] = top_left

                tmp_events[:, :2] = (tmp_events[:, :2] - c_xy) * bbox_scale + c_xy
                tmp_events[:, :2] -= top_left
                mask_bbox = remove_unfeasible_events(tmp_events, input_size[0], input_size[1])
                tmp_events = tmp_events[mask_bbox]

            # if the pixel is not integer
            interpolate = self.config['preprocess']['interpolate']
            output_tmp['crop_eci'][i] = event_to_channels(tmp_events,input_size[0],
                                            input_size[1],
                                            interpolate=interpolate)
            # encode events for optical flow
            if self.config['method']['flow']['usage']:
                flow_repre = event_to_voxel(tmp_events,
                                            self.config['method']['flow']['num_bins'],
                                            input_size[0],
                                            input_size[1],
                                            interpolate)
                output_tmp['flow_repre'][i] = flow_repre.reshape(-1, input_size[0], input_size[1])

            # spilt the events in [0,1] to some bins
            time_markers = torch.arange(self.config['method']['event_encoder']['num_bins'])
            time_indices_tmp = torch.searchsorted(tmp_events[:, 3]*self.config['method']['event_encoder']['num_bins'], time_markers)
            start_indices_tmp = time_indices_tmp
            end_indices_tmp = torch.zeros_like(time_indices_tmp)
            end_indices_tmp[:self.config['method']['event_encoder']['num_bins']-1] = time_indices_tmp[1:]
            end_indices_tmp[-1] = tmp_events.shape[0]-1
            for j in range(self.config['method']['event_encoder']['num_bins']):
                tmp_events_encoder = tmp_events[start_indices_tmp[j]:end_indices_tmp[j]].clone()
                tmp_events_encoder[:, 3] = tmp_events_encoder[:, 3] * self.config['method']['event_encoder']['num_bins'] - time_markers[j]
                # LNES representation
                if self.config['preprocess']['repre'] == 'LNES':
                    lnes = event_to_LNES(tmp_events_encoder,
                                         input_size[0],
                                         input_size[1], interpolate=interpolate)
                    if self.mode == 'train':
                        lnes = self.add_noise(lnes,
                                            self.config['preprocess']['augment']['gaussian_mean'],
                                            self.config['preprocess']['augment']['gaussian_sigma'],
                                            self.config['preprocess']['augment']['salt_pepper'])
                    output_tmp['event_encoder_repre'][i, j*2:j*2+2] = lnes

            # make the event padded
            if unbbox_events.shape[0] > self.config['preprocess']['num_events']:
                select_indices = torch.randperm(unbbox_events.shape[0])[
                                 :self.config['preprocess']['num_events']].sort().values
                unbbox_events = unbbox_events[select_indices]
            else:
                unbbox_events = torch.cat(
                    [unbbox_events, torch.zeros(self.config['preprocess']['num_events'] - output_tmp['Num_events'][i], 4)], dim=0)
            img_pol_mask = create_polarity_mask(unbbox_events[:, 2])
            output_tmp['events'][i] = unbbox_events
            output_tmp['pol_mask'][i] = img_pol_mask
            output_tmp['eci'][i] = event_to_channels(unbbox_events,height=self.config['data']['height'], width=self.config['data']['width'])

        return output_tmp



class EvHandDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.mode = config['exper']['mode']
        self.mano_key = self.config['data']['mano_key']
        self.load_annotations()
        self.load_events()
        self.process_annotations()
        if self.config['preprocess']['acquire_aps']:
            self.process_aps()
        self.regenerate_items()
        if self.annoted:
            self.get_joints_matrix()
        else:
            self.get_bbox_matrix()

    def load_annotations(self):
        annot_path = os.path.join(self.config['data']['seq_dir'], 'annot.json')
        if not os.path.exists(annot_path):
            self.len = 0
            raise ValueError('No annotation.json file at ' + annot_path)
        else:
            self.annot = json_read(annot_path)
            self.hand_type = self.annot['hand_type']
            self.time_sync = self.annot['delta_time']
            self.scene = self.annot['scene']
            self.motion_type = self.annot['motion_type']
            self.gesture_type = self.annot['gesture_type']
            self.subject_id = self.annot['subject_id']
            # self.annoted = self.annot['annoted']
            # apart from the fast motion sequences, other sequences has 3d annotations
            self.annoted = self.motion_type != 'fast'
            # get the distortion parameters
            K_old = np.array(self.annot['camera_info']['event']['K_old'])
            K_new = np.array(self.annot['camera_info']['event']['K_new'])
            dist = np.array(self.annot['camera_info']['event']['dist'])
            self.undistortion_param = [K_old, dist, K_new]
            # get the time_window
            if self.config['preprocess']['same_time_window'] is not None:
                self.time_window = self.config['preprocess']['same_time_window']
            else:
                self.time_window = 1000000. / self.config['data']['fps']
            if self.config['data']['flip'] and self.hand_type == 'left':
                self.left_flip = True

    def load_events(self):
        filenames = os.listdir(self.config['data']['seq_dir'])
        event_file = None
        for filename in filenames:
            if filename.endswith('.aedat4'):
                event_file = os.path.join(self.config['data']['seq_dir'], filename)
                break
        if event_file is None:
            raise ValueError('No aedat4 file at ' + self.config['data']['seq_dir'])
        # get events, aps and trigger info from aedat4 file
        events, frames, triggers = extract_data_from_aedat4(event_file)
        self.events = np.vstack([events['x'], events['y'], events['polarity'], events['timestamp']]).T
        if self.config['preprocess']['acquire_aps']:
            self.frames = frames


        # sort with timestamp
        if not np.all(np.diff(self.events[:, 3]) >= 0):
            self.events = self.events[np.argsort(self.events[:, 3])]
        # set the first event timestamp to 0
        self.first_event_time = self.events[0, 3]
        # warning!
        # the timestamp datatype of event are int64, if you convert it to np.float32 directly, you'll lose time accuracy
        self.events[:, 3] = self.events[:, 3] - self.first_event_time
        self.events = self.events.astype(np.float32)
        # undistortion the events
        if self.annot is not None:
            self.events[:, :2], legal_indices = undistortion_points(self.events[:, :2], self.undistortion_param[0],
                                                                    self.undistortion_param[1], self.undistortion_param[2],
                                                                    set_bound=True, width=self.config['data']['width'],
                                                                    height=self.config['data']['height'])
            self.events = self.events[legal_indices]

        if self.left_flip:
            self.events[:, :1] = 2 * self.annot['camera_info']['event']['K'][0][2] - self.events[:, :1]
        # filter the noise by background filter
        if self.config['preprocess']['activity_filter']:
            is_noise = background_activity_filter(self.events,
                                                 size=(self.height, self.width),
                                                 delta_t=self.config['preprocess']['activity_filter_delta_t'],
                                                 N_thre=self.config['preprocess']['activity_filter_N_thre'])
            self.events = self.events[is_noise]

    def process_aps(self):
        # use the aps from the DAVIS346
        self.aps_info = []
        for aps in self.frames:
            if self.undistortion_param is None:
                aps_undistorted = aps.image
            else:
                aps_undistorted = cv2.undistort(aps.image, self.undistortion_param[0], self.undistortion_param[1],
                                                None, self.undistortion_param[2])
            self.aps_info.append({'timestamp': float(aps.timestamp - self.first_event_time),
                                  'image': aps_undistorted})

    def process_annotations(self):
        # get all the available annotations
        if self.left_flip:
            mano_ids = [int(id) for id in self.annot[self.mano_key+'_flip'].keys()]
        else:
            mano_ids = [int(id) for id in self.annot[self.mano_key].keys()]
        joints_3d_ids = [int(id) for id in self.annot['3d_joints'].keys()]
        mano_ids.sort()
        joints_3d_ids.sort()
        ids = []
        # if not fast motion sequences, get the 3D annotation as label
        for id in mano_ids:
            if id in joints_3d_ids:
                ids.append(id)
        if self.mode == 'eval':
            # for fast motion, we evaluate it with 2D joints
            if self.motion_type == 'fast':
                for key in self.annot['2d_joints']['event'].keys():
                    if self.annot['2d_joints']['event'][key] != []:
                        if len(self.annot['2d_joints']['event'][key]) == 21:
                            ids.append(int(key))
                ids.sort()
            # fps rate relative to the origin annotation fps
            if self.motion_type == 'fast':
                rate = self.config['preprocess']['test_fps_fast'] / float(self.config['data']['fps'])
            else:
                rate = self.config['preprocess']['test_fps'] / float(self.config['data']['fps'])
            if not rate.is_integer():
                raise ValueError('Unavailable fps setting for data and test!')
            else:
                for id in np.arange(0, ids[-1], 1./rate):
                    if id.is_integer() and int(id) in ids:
                        self.annot_info.append({
                            'image_id': str(int(id)),
                            'timestamp': self.time_sync + id * 1000000. / self.config['data']['fps']
                        })
                    else:
                        # if there is no annotation, the annotation id is set to -1
                        self.annot_info.append({
                            'image_id': '-1',
                            'timestamp': self.time_sync + id * 1000000. / self.config['data']['fps'],
                        })
        elif self.mode == 'val' and not self.annot['annoted']:
            # we don't evaluate the sequences without manual check
            pass
        else:
            for id in ids:
                self.annot_info.append({
                    'image_id': str(id),
                    'timestamp': self.time_sync + int(id)*1000000. / self.config['data']['fps']
                })
        pass

    def get_annotations(self, id):
        '''
        get annotation at image id
        including mano parameters, camera intrinsics and extrinsics, 3D annotations
        '''

        annot = {}
        annot['hand_type'] = 1 if self.hand_type == 'right' else 0
        # index : sequence index
        annot['index'] = self.index
        if id == '-1' or self.motion_type == 'fast':
            annot['mano_rot_pose'] = torch.zeros((3, ), dtype=torch.float32)
            annot['mano_hand_pose'] = torch.zeros((45, ), dtype=torch.float32)
            annot['mano_shape'] = torch.zeros((10, ), dtype=torch.float32)
            annot['mano_trans'] = torch.zeros((3, ), dtype=torch.float32)
            annot['joint'] = torch.zeros((21, 3), dtype=torch.float32)
            annot['root_joint'] = torch.zeros((3, ), dtype=torch.float32)
        else:
            if self.left_flip:
                mano = self.annot[self.mano_key + '_flip'][id]
            else:
                mano = self.annot[self.mano_key][id]
            annot['mano_rot_pose'] = torch.tensor(mano['rot'], dtype=torch.float32).view(-1)
            annot['mano_hand_pose'] = torch.tensor(mano['hand_pose'], dtype=torch.float32).view(-1)
            annot['mano_shape'] = torch.tensor(mano['shape'], dtype=torch.float32).view(-1)
            annot['mano_trans'] = torch.tensor(mano['trans'], dtype=torch.float32).view(3)
            # change joint indices from evhand to smplx
            annot['joint'] = torch.tensor(self.annot['3d_joints'][str(id)], dtype=torch.float32).view(-1, 3)[
                indices_change(2, 1)] / 1000.
            annot['root_joint'] = torch.tensor(mano['root_joint'], dtype=torch.float32).view(3)
        annot['joint_valid'] = torch.ones((21, 1), dtype=torch.float32)[indices_change(2, 1)]
        # add camera intrinsics and extrinsics
        annot['R'] = torch.tensor(self.annot['camera_info']['event']['R'], dtype=torch.float32).view(3, 3)
        annot['K'] = torch.tensor(self.annot['camera_info']['event']['K'], dtype=torch.float32).view(3, 3)
        t = torch.tensor(self.annot['camera_info']['event']['T'], dtype=torch.float32).view(3) / 1000.
        annot['t'] = t
        # transport to camera coordinate systems for augmentation
        # in smplx, the global trans and rot are not purely global transform, there is a root joint trans vector needed
        # for this transform
        # we want to transform the mano parameters to the camera coordinate
        R_tmp = roma.rotvec_to_rotmat(annot['mano_rot_pose'])
        t_tmp = annot['mano_trans']
        R_tmp_new = torch.mm(annot['R'], R_tmp)
        rot_pose_new = roma.rotmat_to_rotvec(R_tmp_new)
        trans_new = torch.mm(annot['R'], (t_tmp + annot['root_joint'])[:, None])[:, 0] + t - annot['root_joint']
        joint_new = (torch.mm(annot['R'], annot['joint'].transpose(0, 1)) + t[:, None]).transpose(0, 1)
        if self.left_flip:
            joint_new[:, 0:1] = -joint_new[:, 0:1]
        annot['joint'] = joint_new
        annot['mano_rot_pose'] = rot_pose_new
        annot['mano_trans'] = trans_new
        annot['R'] = torch.eye(3)
        annot['t'] = torch.zeros(3)
        annot['2d_joints'] = torch.zeros((21, 2), dtype=torch.float32)
        annot['fast'] = 0
        if self.motion_type == 'fast':
            annot['fast'] = 1
        if self.motion_type == 'fast' and id != '-1':
            joints_2d = np.array(self.annot['2d_joints']['event'][id], dtype=np.float32).reshape(21, 2)
            # get the undistorted 2D joints
            joints_2d, legal_indices = undistortion_points(joints_2d.reshape(-1, 2), self.undistortion_param[0],
                                                        self.undistortion_param[1], self.undistortion_param[2],
                                                        set_bound=True, width=self.config['data']['width'],
                                                        height=self.config['data']['height'])
            annot['2d_joints'] = torch.tensor(joints_2d, dtype=torch.float32).reshape(-1, 2)# [indices_change(2, 1)]
            if self.left_flip:
                annot['2d_joints'][:, 0:1] = 2 * self.annot['camera_info']['event']['K'][0][2] - annot['2d_joints'][:, 0:1]
        return annot

    def get_joints_matrix(self):
        '''
        get 3D joints, and use them to interpolate 2D bbox at any time
        '''
        joints_ids = [int(id) for id in self.annot['3d_joints'].keys()]
        joints_ids.sort()
        # a tuple (timestamps, 3d joints)
        self.joints_3d_bbox = [np.zeros((len(joints_ids)), dtype=np.float32), np.zeros((len(joints_ids), 21, 3), dtype=np.float32)]
        ids_ = np.array(joints_ids, dtype=np.float32)
        ids_timestamp = ids_ * 1000000. / self.config['data']['fps'] + self.time_sync
        self.joints_3d_bbox[0] = ids_timestamp
        R = np.array(self.annot['camera_info']['event']['R'], dtype=np.float32).reshape(3, 3)
        t = np.array(self.annot['camera_info']['event']['T'], dtype=np.float32).reshape(1, 3) / 1000.
        for i, id in enumerate(joints_ids):
            self.joints_3d_bbox[1][i] = np.array(self.annot['3d_joints'][str(id)], dtype=np.float32).reshape(-1, 3)[indices_change(2, 1)] / 1000.
        self.joints_3d_bbox[1] = np.einsum('ijk,ikn->ijn', R[None, ...], self.joints_3d_bbox[1].transpose(0, 2, 1)).transpose(0, 2, 1) + t[None, ...]
        if self.left_flip:
            self.joints_3d_bbox[1][:, :, 0:1] = - self.joints_3d_bbox[1][:, :, 0:1]
        # this is 3d joints bbox interpolation function
        self.bbox_inter_f = interp1d(self.joints_3d_bbox[0], self.joints_3d_bbox[1], axis=0, kind='cubic')

    def get_bbox_matrix(self):
        '''
        # get bbox matrix for fast motion sequences
        bbox_seq: (timestamp, center_x, center_y, size)
        '''
        bbox_ids = []
        for key in self.annot['bbox']['event'].keys():
            if self.annot['bbox']['event'][key] != []:
                bbox_ids.append(int(key))
        bbox_ids.sort()
        self.bbox_seq = np.zeros((len(bbox_ids), 4), dtype=np.float32)
        self.bbox_seq[:, 0] = np.array(bbox_ids, dtype=np.float32) * 1000000. / self.config['data']['fps'] + self.time_sync
        # if the bbox is manual annotation, use it as GT bbox; else use machine annotation bbox
        for i, id in enumerate(bbox_ids):
            if str(id) in self.annot['2d_joints']['event'].keys() and self.annot['2d_joints']['event'][str(id)] != []:
                kps2d = np.array(self.annot['2d_joints']['event'][str(id)], dtype=np.float32)
                top = kps2d[:, 1].min()
                bottom = kps2d[:, 1].max()
                left = kps2d[:, 0].min()
                right = kps2d[:, 0].max()
                top_left = np.array([left, top])
                bottom_right = np.array([right, bottom])
                center = (top_left + bottom_right) / 2.0
                bbox_size = np.abs(bottom_right - top_left).max() * 1.5 # * self.config['preprocess']['bbox']['joints_to_bbox_rate']
                center, legal_indices = undistortion_points(center.reshape(-1, 2), self.undistortion_param[0],
                                                                    self.undistortion_param[1], self.undistortion_param[2],
                                                                    set_bound=True, width=self.config['data']['width'],
                                                                    height=self.config['data']['height'])
                center = center[0]
            else:
                top_left = np.array(self.annot['bbox']['event'][str(id)]['tl'], dtype=np.float32)
                bottom_right = np.array(self.annot['bbox']['event'][str(id)]['br'], dtype=np.float32)
                center = (top_left + bottom_right) / 2.0
                bbox_size = np.abs(bottom_right - top_left).max() # * self.config['preprocess']['bbox']['joints_to_bbox_rate'] / 1.5
            if self.left_flip:
                center[0] = 2 * self.annot['camera_info']['event']['K'][0][2] - center[0]
            self.bbox_seq[i, 1:3] = center
            self.bbox_seq[i, 3:4] = bbox_size

        self.bbox_inter_f = interp1d(self.bbox_seq[:, 0], self.bbox_seq[:, 1:4], axis=0, kind='quadratic')

    def __len__(self):
        return self.len