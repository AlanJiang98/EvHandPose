#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import *
import numpy as np
import roma
import matplotlib.pyplot as plt
from core.model.smplx.body_models import MANO
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.utils import cameras_from_opencv_projection
from core.model.utils.iwe import get_interpolation, interpolate
# from tools.visualization.vis_flow import *
from core.model.utils.iwe import purge_unfeasible
import gc

class SupervisionBranch(nn.Module):
    def __init__(self, config):
        super(SupervisionBranch, self).__init__()
        self.config = config
        self.width = self.config['data']['width']
        self.height = self.config['data']['height']
        smplx_path = self.config['data']['smplx_path']
        layers = {
            'right': MANO(smplx_path, use_pca=False, is_rhand=True),
            'left': MANO(smplx_path, use_pca=False, is_rhand=False)}
        if torch.sum(torch.abs(layers['left'].shapedirs[:, 0, :] - layers['right'].shapedirs[:, 0, :])) < 1:
            layers['left'].shapedirs[:, 0, :] *= -1
        if self.config['data']['flip']:
            self.mano_layer = layers['right']
        else:
            self.mano_layer = layers[self.config['data']['hand_type']]
        self.annot_len = self.config['method']['seq_model']['annot_len']
        self.seq_len = self.config['method']['seq_model']['seq_len']
        self.get_render_param()

    def get_render_param(self):
        self.raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            faces_per_pixel=2,
            perspective_correct=True,
            blur_radius=self.config['loss']['basic']['blur_radius'],
        )
        self.lights = PointLights(
            location=[[0, 2, 0]],
            diffuse_color=((0.5, 0.5, 0.5),),
            specular_color=((0.5, 0.5, 0.5),)
        )
        self.render = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=self.raster_settings),
            shader=SoftPhongShader(lights=self.lights)
        )

    def render_mesh(self, manos, eci):
        batch_size, seq_len, _ = manos['mano_rot_pose'].shape
        output = self.mano_layer(
            global_orient=manos['mano_rot_pose'].reshape(-1, 3),
            hand_pose=manos['mano_hand_pose'].reshape(-1, 45),
            betas=manos['mano_shape'].reshape(-1, 10),
            transl=manos['mano_trans'].reshape(-1, 3)
        )
        now_vertices = torch.bmm(self.R[:, -seq_len:].reshape(-1, 3, 3), output.vertices.transpose(2, 1)).transpose(2, 1) + self.t[:, -seq_len:].reshape(-1, 1, 3)
        faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(batch_size*seq_len, 1, 1).type_as(manos['mano_trans'])
        verts_rgb = torch.ones_like(self.mano_layer.v_template).type_as(manos['mano_trans'])
        verts_rgb = verts_rgb.expand(batch_size*seq_len, verts_rgb.shape[0], verts_rgb.shape[1])
        textures = TexturesVertex(verts_rgb)
        mesh = Meshes(
            verts=now_vertices,
            faces=faces,
            textures=textures
        )
        cameras = cameras_from_opencv_projection(
            R=torch.eye(3).repeat(batch_size * seq_len, 1, 1).type_as(manos['mano_shape']),
            tvec=torch.zeros(batch_size * seq_len, 3).type_as(manos['mano_shape']),
            camera_matrix=self.K[:, -seq_len:].reshape(-1, 3, 3).type_as(manos['mano_shape']),
            image_size=torch.tensor([self.height, self.width]).expand(batch_size * seq_len, 2).type_as(manos['mano_shape'])
        ).to(manos['mano_trans'].device)
        self.render.shader.to(manos['mano_trans'].device)
        res = self.render(
            mesh,
            cameras=cameras
        )
        img = res[..., :3]
        img = img.reshape(batch_size, seq_len, self.height, self.width, 3)
        if self.config['log']['render_background']:
            zero_pad = torch.zeros((batch_size, seq_len, self.height, self.width, 1), dtype=torch.float32).type_as(img)
            eci_tmp = torch.cat([eci.clone(), zero_pad], dim=-1)
            mask = res[..., 3:4].reshape(batch_size, seq_len, self.height, self.width, 1) != 0.
            img = torch.clip(img * mask + mask.logical_not() * eci_tmp, 0, 1)
        return img

    def forward_flow(self, manos):
        batch_size, seq_len, _ = manos['mano_hand_pose'].shape
        N_inter = self.config['loss']['cm']['N_inter']
        coeff_inter = torch.linspace(0., 1., N_inter).type_as(manos['mano_shape'])

        rot_inter = roma.rotvec_slerp(
            manos['mano_rot_pose'][:, :-1].reshape(-1, 3),
            manos['mano_rot_pose'][:, 1:].reshape(-1, 3),
            coeff_inter
        ).reshape(-1, 3)

        # print('rot_inter', rot_inter.reshape(N_inter, batch_size, seq_len-1, -1))
        hand_pose_inter = roma.rotvec_slerp(
            manos['mano_hand_pose'][:, :-1].reshape(-1, 3),
            manos['mano_hand_pose'][:, 1:].reshape(-1, 3),
            coeff_inter
        ).reshape(-1, 45)

        # interpolate the mano parameter
        coeff_inter_repeat = coeff_inter.repeat((1, 1, 1, 1)).permute(3, 1, 2, 0)
        shape_inter = manos['mano_shape'][:, 1:].repeat((N_inter, 1, 1, 1)) * coeff_inter_repeat +\
                      manos['mano_shape'][:, :-1].repeat((N_inter, 1, 1, 1)) * (1. - coeff_inter_repeat)
        trans_inter = manos['mano_trans'][:, 1:].repeat((N_inter, 1, 1, 1)) * coeff_inter_repeat +\
                        manos['mano_trans'][:, :-1].repeat((N_inter, 1, 1, 1)) * (1. - coeff_inter_repeat)
        shape_inter = shape_inter.reshape(-1, 10)
        trans_inter = trans_inter.reshape(-1, 3)

        output = self.mano_layer(
            global_orient=rot_inter,
            hand_pose=hand_pose_inter,
            betas=shape_inter,
            transl=trans_inter
        )

        vertices_inter = torch.bmm(self.R[:, 1:].reshape(-1, 3, 3).repeat(N_inter, 1, 1),
                                   output.vertices.transpose(2, 1)).transpose(2, 1)\
                                + self.t[:, 1:].reshape(-1, 1, 3).repeat(N_inter, 1, 1)

        kps_inter = torch.bmm(self.K[:, 1:].reshape(-1, 3, 3).repeat(N_inter, 1, 1),
                              vertices_inter.transpose(2, 1)).transpose(2, 1)
        kps_inter = torch.div(kps_inter, kps_inter[:, :, 2:])[..., :2]
        kps_inter = kps_inter.reshape(N_inter, batch_size, seq_len-1, -1, 2)
        # compute the vertex speed
        flow_speed_fw = (kps_inter[-1:] - kps_inter[:-1]) / (
                1. - coeff_inter.repeat(1, 1, 1, 1, 1).permute(4, 1, 2, 3, 0)[:-1])
        flow_speed_fw = torch.cat([flow_speed_fw, flow_speed_fw[-1:]], dim=0)

        faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(N_inter * batch_size * (seq_len-1), 1, 1).type_as(manos['mano_shape'])
        cameras = cameras_from_opencv_projection(
            R=torch.eye(3).repeat(N_inter * batch_size * (seq_len-1), 1, 1).type_as(manos['mano_shape']),
            tvec=torch.zeros(N_inter * batch_size * (seq_len-1), 3).type_as(manos['mano_shape']),
            camera_matrix=self.K[:, 1:].reshape(-1, 3, 3).repeat(N_inter, 1, 1).type_as(manos['mano_shape']),
            image_size=torch.tensor([self.height, self.width]).expand(N_inter * batch_size * (seq_len-1), 2).type_as(manos['mano_shape'])
        ).to(manos['mano_shape'].device)
        verts_rgb = torch.ones_like(self.mano_layer.v_template).repeat(N_inter * batch_size * (seq_len-1), 1, 1).type_as(manos['mano_shape'])
        textures = TexturesVertex(verts_rgb)

        # set the verts requires_grad=False, or will cause the gradient exploding
        mesh = Meshes(
            verts=vertices_inter.detach(),
            faces=faces,
            textures=textures
        )
        frags = self.render.rasterizer(mesh, cameras=cameras)

        faces_tmp = faces.reshape(-1, 3)
        mask = frags.pix_to_face[..., 0] > -1
        verts_index = faces_tmp[frags.pix_to_face[..., 0]]
        flow_x_fw = torch.gather(flow_speed_fw[..., 0], 3,
                              verts_index.reshape(N_inter, batch_size, seq_len-1, -1).long())
        flow_y_fw = torch.gather(flow_speed_fw[..., 1], 3,
                              verts_index.reshape(N_inter, batch_size, seq_len-1, -1).long())
        flow_x_fw = flow_x_fw.reshape(N_inter, batch_size, seq_len-1, self.height, self.width, 3)
        flow_y_fw = flow_y_fw.reshape(N_inter, batch_size, seq_len-1, self.height, self.width, 3)
        flow_xy_fw = torch.stack([flow_x_fw, flow_y_fw], dim=-1)

        # get the mean flow
        bary_flow_weights = frags.bary_coords[:, :, :, 0, :].reshape(N_inter, batch_size, seq_len-1,
                                                                     self.height, self.width, 3, 1)
        mask = mask.reshape(N_inter, batch_size, seq_len-1, self.height, self.width, 1)
        flows_fw = torch.sum(flow_xy_fw * bary_flow_weights * mask[..., None, :], dim=-2)
        weight = mask
        flow_fw = torch.sum(flows_fw, dim=0) / (torch.sum(weight, dim=0) + 1e-7)
        mask = torch.sum(mask, dim=0) > 0

        return flow_fw, mask

    def compute_flow_a_2_b(self, pre_vertices, now_vertices, batch_len):
        normals_now = compute_normals(now_vertices, self.mano_layer.faces)
        normals_pre = compute_normals(pre_vertices, self.mano_layer.faces)
        kps_now = torch.bmm(self.K[:, 1:].reshape(-1, 3, 3), now_vertices.transpose(2, 1)).transpose(2, 1)
        kps_pre = torch.bmm(self.K[:, :-1].reshape(-1, 3, 3), pre_vertices.transpose(2, 1)).transpose(2, 1)
        view_vertor_now = F.normalize(now_vertices, eps=1e-6, dim=2)
        view_vertor_pre = F.normalize(pre_vertices, eps=1e-6, dim=2)
        cos_now = torch.sum(torch.mul(normals_now, view_vertor_now), dim=2)
        cos_pre = torch.sum(torch.mul(normals_pre, view_vertor_pre), dim=2)
        weights_now = torch.sigmoid(5 * torch.relu(1 - cos_now + 0.2))
        weights_pre = torch.sigmoid(5 * torch.relu(1 - cos_pre + 0.2))
        weights_face = weights_now * weights_pre
        kps_now = torch.div(kps_now, kps_now[:, :, 2:])
        kps_pre = torch.div(kps_pre, kps_pre[:, :, 2:])
        flow_kps = kps_now - kps_pre
        flow_kps = torch.cat([flow_kps[:, :, :2] for i in range(4)], dim=1)
        mask_now = (kps_now[:, :, 0] < self.width) * (kps_now[:, :, 0] >= 0) * (kps_now[:, :, 1] < self.height) * (
                    kps_now[:, :, 1] >= 0)
        mask_pre = (kps_pre[:, :, 0] < self.width) * (kps_pre[:, :, 0] >= 0) * (kps_pre[:, :, 1] < self.height) * (
                    kps_pre[:, :, 1] >= 0)
        mask_inplane = mask_now * mask_pre
        weights_face = (weights_face * mask_inplane).detach()
        weights_face = torch.cat([weights_face for i in range(4)], dim=1)
        top_y = torch.floor(kps_pre[:, :, 1:2])
        bot_y = torch.floor(kps_pre[:, :, 1:2] + 1)
        left_x = torch.floor(kps_pre[:, :, 0:1])
        right_x = torch.floor(kps_pre[:, :, 0:1] + 1)

        top_left = torch.cat([left_x, top_y], dim=2)
        top_right = torch.cat([right_x, top_y], dim=2)
        bottom_left = torch.cat([left_x, bot_y], dim=2)
        bottom_right = torch.cat([right_x, bot_y], dim=2)
        idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        kps_pre_4 = torch.cat([kps_pre[:, :, :2] for i in range(4)], dim=1)
        zeros = torch.zeros(kps_pre_4.shape).type_as(pre_vertices)
        weights_bi = torch.max(zeros, 1 - torch.abs(kps_pre_4 - idx))
        # purge unfeasible indices
        idx, mask = purge_unfeasible(idx, (self.height, self.width))
        weights_bi = torch.prod(weights_bi, dim=-1, keepdim=True) * mask  # bilinear interpolation
        idx[:, :, 1] *= self.width  # torch.view is row-major
        idx = torch.sum(idx, dim=2, keepdim=True).detach()
        weights = (weights_face[..., None] * weights_bi).detach()

        flow_weights = torch.zeros((batch_len, self.height * self.width, 1)).type_as(pre_vertices)
        flow_weights = flow_weights.scatter_add_(1, idx.long(), weights)
        flow_weights = flow_weights.view((batch_len, self.height, self.width, 1))

        flow_x = torch.zeros((batch_len, self.height * self.width, 1)).type_as(pre_vertices)
        flow_x = flow_x.scatter_add_(1, idx.long(), weights * flow_kps[:, :, 0:1])
        flow_x = flow_x.view((batch_len, self.height, self.width, 1))
        flow_y = torch.zeros((batch_len, self.height * self.width, 1)).type_as(pre_vertices)
        flow_y = flow_y.scatter_add_(1, idx.long(), weights * flow_kps[:, :, 1:2])
        flow_y = flow_y.view((batch_len, self.height, self.width, 1))
        flow_all = torch.cat([flow_x, flow_y], dim=3)

        return flow_all, flow_weights

    def warp(self, flow, time_marker, events, events_mask, events_indices, scale=1.0):
        batch_size, seq_len, _, _, _ = flow.shape
        flow = flow.reshape(-1, self.height, self.width, 2)
        events = events[:, -seq_len:].reshape(-1, events.shape[2], 4)
        events_indices = events_indices.reshape(-1, 4)
        events_mask = events_mask.reshape(-1, events_mask.shape[2], 2)
        fw_mask = torch.zeros(events_mask.shape).type_as(events)
        bw_mask = torch.zeros(events_mask.shape).type_as(events)
        for i in range(events_mask.shape[0]):
            fw_mask[i, events_indices[i, 0]:events_indices[i, 1]] = 1
            bw_mask[i, events_indices[i, 2]:events_indices[i, 3]] = 1

        fw_events_mask = torch.cat([fw_mask * events_mask for i in range(4)], dim=1)
        bw_events_mask = torch.cat([bw_mask * events_mask for i in range(4)], dim=1)

        flow_idx = events[:, :, :2].long().clone() # crazy bug if flow_idx is not integer
        flow_idx[:, :, 1] *= self.width
        flow_idx = torch.sum(flow_idx, dim=2)

        flow = flow.view(flow.shape[0], -1, 2)
        event_flowy = torch.gather(flow[:, :, 1], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, :, 0], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowx, event_flowy], dim=2)


        fw_idx, fw_weights = get_interpolation(events, event_flow, time_marker[1], (self.height, self.width), scale)
        # per-polarity image of (forward) warped events
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, (self.height, self.width),
                                 polarity_mask=fw_events_mask[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, (self.height, self.width),
                                 polarity_mask=fw_events_mask[:, :, 1:2])

        bw_idx, bw_weights = get_interpolation(events, event_flow, time_marker[0], (self.height, self.width), scale)
        # per-polarity image of (forward) warped events
        bw_iwe_pos = interpolate(bw_idx.long(), bw_weights, (self.height, self.width),
                                 polarity_mask=bw_events_mask[:, :, 0:1])
        bw_iwe_neg = interpolate(bw_idx.long(), bw_weights, (self.height, self.width),
                                 polarity_mask=bw_events_mask[:, :, 1:2])
        fw_iwe = torch.cat([fw_iwe_pos, fw_iwe_neg], dim=-1)
        bw_iwe = torch.cat([bw_iwe_pos, bw_iwe_neg], dim=-1)
        return fw_iwe, bw_iwe

    def get_event_slice(self, factor, event_nums):
        real_event_num = torch.min(
            event_nums,
            self.config['preprocess']['num_events'] * torch.ones_like(event_nums)
        )
        event_nums = torch.max(factor*real_event_num, 120 * torch.ones_like(real_event_num))
        event_indices = torch.zeros(event_nums.shape[0], event_nums.shape[1], 4).type_as(event_nums)
        event_indices[:, :, 0] = real_event_num - event_nums
        event_indices[:, :, 1] = real_event_num * 1
        event_indices[:, :, 2] = real_event_num * 0
        event_indices[:, :, 3] = event_nums
        return event_indices.long()

    def get_cm_loss(self, iwe_fw, iwe_bw):
        N_sum_fw = torch.sum((iwe_fw - torch.mean(iwe_fw, dim=[1, 2], keepdim=True)) ** 2, dim=[1, 2, 3])
        N_sum_bw = torch.sum((iwe_bw - torch.mean(iwe_bw, dim=[1, 2], keepdim=True)) ** 2, dim=[1, 2, 3])
        N_div = iwe_fw.shape[1] * iwe_fw.shape[2] * iwe_fw.shape[3] + 1e-8
        loss_fw = N_sum_fw / N_div
        loss_bw = N_sum_bw / N_div
        return loss_fw.mean(), loss_bw.mean()

    def get_smooth_loss(self, manos):
        trans_loss = F.mse_loss(
            manos['mano_trans'][:, :-1],
            manos['mano_trans'][:, 1:]
        )
        rot_loss = F.mse_loss(
            manos['mano_rot_pose'][:, :-1],
            manos['mano_rot_pose'][:, 1:]
        )
        hand_pose_loss = F.mse_loss(
            manos['mano_hand_pose'][:, :-1],
            manos['mano_hand_pose'][:, 1:]
        )
        shape_loss = F.mse_loss(
            manos['mano_shape'][:, :-1],
            manos['mano_shape'][:, 1:]
        )
        smooth_loss_origin = self.config['loss']['smooth']['trans_weight'] * trans_loss \
                    + self.config['loss']['smooth']['rot_weight'] * rot_loss \
                    + self.config['loss']['smooth']['hand_pose_weight'] * hand_pose_loss \
                    + self.config['loss']['smooth']['shape_weight'] * shape_loss
        if self.config['loss']['smooth']['soften']:
            smooth_loss = torch.relu(smooth_loss_origin - torch.tensor(self.config['loss']['smooth']['margin']))
            smooth_loss = self.config['loss']['smooth']['soften_rate'] * torch.sigmoid(smooth_loss)
        else:
            smooth_loss = torch.max(smooth_loss_origin, torch.tensor(self.config['loss']['smooth']['margin']))
        if self.config['log']['verbose']:
            print('smooth: ', trans_loss, rot_loss, hand_pose_loss, shape_loss)
        return smooth_loss, smooth_loss_origin

    def get_edge_loss(self, iwe_fw, iwe_bw, manos):
        '''
        we compute forward and backwork edge loss
        '''
        batch_size, seq_len, _ = manos['mano_hand_pose'].shape
        num_manos = batch_size*seq_len
        num_pairs = batch_size*(seq_len-1)
        num_verts = self.mano_layer.get_num_verts()
        output = self.mano_layer(
            global_orient=manos['mano_rot_pose'].reshape(-1, 3),
            hand_pose=manos['mano_hand_pose'].reshape(-1, 45),
            betas=manos['mano_shape'].reshape(-1, 10),
            transl=manos['mano_trans'].reshape(-1, 3)
        )
        now_vertices = torch.bmm(self.R[:, -seq_len:].reshape(-1, 3, 3), output.vertices.transpose(2, 1)).transpose(2, 1)\
                       + self.t[:, -seq_len:].reshape(-1, 1, 3)

        now_vertices = now_vertices.reshape(batch_size, seq_len, -1, 3)

        # get 2d kps
        kps_now = torch.bmm(self.K[:, -seq_len+1:].reshape(-1, 3, 3),
                            now_vertices[:, 1:].reshape(-1, num_verts, 3).transpose(2, 1)).transpose(2, 1)
        kps_pre = torch.bmm(self.K[:, -seq_len:-1].reshape(-1, 3, 3),
                            now_vertices[:, :-1].reshape(-1, num_verts, 3).transpose(2, 1)).transpose(2, 1)
        mask_boundary_fw = (kps_now[:, :, 0] < self.width) * (kps_now[:, :, 0] >= 0) * (kps_now[:, :, 1] < self.height) * (kps_now[:, :, 1] >= 0)
        mask_boundary_bw = (kps_pre[:, :, 0] < self.width) * (kps_pre[:, :, 0] >= 0) * (kps_pre[:, :, 1] < self.height) * (kps_pre[:, :, 1] >= 0)
        kps_now = torch.div(kps_now, kps_now[:, :, 2:])[:, :, :2]
        kps_pre = torch.div(kps_pre, kps_pre[:, :, 2:])[:, :, :2]
        # add motion weight
        if self.config['loss']['edge']['motion']:
            flow_kps = kps_now - kps_pre
            flow_kps = flow_kps.detach()
            weights_motion = torch.sqrt(torch.sum(flow_kps ** 2, dim=2)) + self.config['loss']['edge']['motion_bias']
        else:
            weights_motion = torch.ones(kps_now.shape[0], num_verts).type_as(iwe_fw)

        # add orientation weight
        faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).type_as(iwe_fw) #.to(device)
        faces = faces.expand(num_manos, faces.shape[0], faces.shape[1])
        verts_rgb = torch.ones_like(self.mano_layer.v_template).type_as(iwe_fw)
        verts_rgb = verts_rgb.expand(num_manos, verts_rgb.shape[0], verts_rgb.shape[1])
        textures = TexturesVertex(verts_rgb)
        mesh = Meshes(
            verts=now_vertices.reshape(batch_size*seq_len, -1, 3).detach(),
            faces=faces,
            textures=textures
        )

        if self.config['loss']['edge']['orient']:
            normals_now = mesh.verts_normals_packed().reshape(num_manos, num_verts, 3)
            view_vector_now = F.normalize(now_vertices.reshape(num_manos, num_verts, 3), eps=1e-6, dim=2)
            cos_now = torch.sum(torch.mul(normals_now, view_vector_now), dim=2)
            cos_now = cos_now.detach()
            weights_orient = self.config['loss']['edge']['orient_bias'] + 1 - torch.abs(cos_now)
        else:
            weights_orient = torch.ones(num_manos, num_verts).type_as(iwe_fw)#.to(device)
        weights_orient = weights_orient.reshape(batch_size, seq_len, -1)

        # weight and orientation
        weights_verts_fw = mask_boundary_fw * weights_orient[:, 1:].reshape(-1, num_verts) * weights_motion
        weights_verts_bw = mask_boundary_bw * weights_orient[:, :-1].reshape(-1, num_verts) * weights_motion

        faces = torch.tensor(self.mano_layer.faces.astype(np.int32)).repeat(num_manos, 1, 1).type_as(iwe_fw)

        # render the edge image
        # render the edge to the plane with weights
        cameras = cameras_from_opencv_projection(
            R=torch.eye(3).repeat(num_manos, 1, 1).type_as(iwe_fw),
            tvec=torch.zeros(num_manos, 3).type_as(iwe_fw),
            camera_matrix=self.K[:, -seq_len:].reshape(-1, 3, 3),
            image_size=torch.tensor([self.height, self.width]).expand(num_manos, 2).type_as(iwe_fw)
        ).to(iwe_fw.device)

        frags = self.render.rasterizer(mesh.detach(), cameras=cameras)
        faces_tmp = faces.reshape(-1, 3)
        verts_index = faces_tmp[frags.pix_to_face[..., 0]].reshape(batch_size, seq_len, -1)

        mask_mesh_edge = frags.pix_to_face[..., 0] > -1
        mask_mesh_edge = mask_mesh_edge.reshape(batch_size, seq_len, self.height, self.width, 1)
        verts_weights_fw_tmp = torch.gather(weights_verts_fw, 1, verts_index[:, 1:].reshape(num_pairs, -1).long())
        verts_weights_fw_tmp = verts_weights_fw_tmp.reshape(-1, self.height, self.width, 3)
        verts_weights_bw_tmp = torch.gather(weights_verts_bw, 1, verts_index[:, :-1].reshape(num_pairs, -1).long())
        verts_weights_bw_tmp = verts_weights_bw_tmp.reshape(-1, self.height, self.width, 3)

        bary_flow_weights = frags.bary_coords[:, :, :, 0, :].reshape(batch_size, seq_len, self.height, self.width, 3)
        mesh_edge_fw = torch.sum(verts_weights_fw_tmp * bary_flow_weights[:, 1:].reshape(-1, self.height, self.width, 3) * mask_mesh_edge[:, 1:].reshape(-1, self.height, self.width, 1), dim=-1)
        mesh_edge_bw = torch.sum(verts_weights_bw_tmp * bary_flow_weights[:, :-1].reshape(-1, self.height, self.width, 3) * mask_mesh_edge[:, :-1].reshape(-1, self.height, self.width, 1), dim=-1)

        # barycoordinates to make the edge image dense and differentiate
        kps_x_fw = torch.gather(kps_now[..., 0], 1, verts_index[:, 1:].reshape(num_pairs, -1).long())
        kps_y_fw = torch.gather(kps_now[..., 1], 1, verts_index[:, 1:].reshape(num_pairs, -1).long())
        kps_now_proj_fw = torch.stack([kps_y_fw, kps_x_fw], dim=-1)
        kps_now_proj_fw = kps_now_proj_fw.reshape(num_pairs, self.height, self.width, 3, 2)
        kps_center_fw = torch.sum(kps_now_proj_fw * bary_flow_weights[:, 1:].reshape(-1, self.height, self.width, 3, 1), dim=-2)

        kps_x_bw = torch.gather(kps_pre[..., 0], 1, verts_index[:, :-1].reshape(num_pairs, -1).long())
        kps_y_bw = torch.gather(kps_pre[..., 1], 1, verts_index[:, :-1].reshape(num_pairs, -1).long())
        kps_now_proj_bw = torch.stack([kps_y_bw, kps_x_bw], dim=-1)
        kps_now_proj_bw = kps_now_proj_bw.reshape(num_pairs, self.height, self.width, 3, 2)
        kps_center_bw = torch.sum(kps_now_proj_bw * bary_flow_weights[:, :-1].reshape(-1, self.height, self.width, 3, 1), dim=-2)

        # select the top eci pixels and try to match the pixel to edge image
        iwe_percentage = self.config['loss']['edge']['iwe_percentage']
        iwe_select_threshold_fw = torch.quantile(iwe_fw.reshape(iwe_fw.shape[0], -1, 2), 1 - iwe_percentage, dim=1).reshape(iwe_fw.shape[0], 1, 1, 2)
        index_iwe_fw = torch.where(iwe_fw > iwe_select_threshold_fw)

        iwe_select_threshold_bw = torch.quantile(iwe_bw.reshape(iwe_bw.shape[0], -1, 2), 1 - iwe_percentage, dim=1).reshape(iwe_bw.shape[0], 1, 1, 2)
        index_iwe_bw = torch.where(iwe_bw > iwe_select_threshold_bw)

        N_selected = (int(iwe_fw.shape[1] * iwe_fw.shape[2] * iwe_percentage) + 1) * 2

        index_iwe_fw_padded = -1 * torch.zeros(num_pairs, N_selected, 2).type_as(iwe_fw)
        index_iwe_bw_padded = -1 * torch.zeros(num_pairs, N_selected, 2).type_as(iwe_fw)

        count_fw = 0
        count_bw = 0
        for i in range(num_pairs):
            index_tmp_fw = index_iwe_fw[0] == i
            num_tmp_fw = index_tmp_fw.sum()
            index_iwe_fw_padded[i, :num_tmp_fw, 0] = index_iwe_fw[1][index_tmp_fw]
            index_iwe_fw_padded[i, :num_tmp_fw, 1] = index_iwe_fw[2][index_tmp_fw]
            count_fw += num_tmp_fw
            index_tmp_bw = index_iwe_bw[0] == i
            num_tmp_bw = index_tmp_bw.sum()
            index_iwe_bw_padded[i, :num_tmp_bw, 0] = index_iwe_bw[1][index_tmp_bw]
            index_iwe_bw_padded[i, :num_tmp_bw, 1] = index_iwe_bw[2][index_tmp_bw]
            count_bw += num_tmp_bw

        mask_index_selected_fw = index_iwe_fw_padded > -1
        index_iwe_fw_padded[torch.logical_not(mask_index_selected_fw)] = 0
        mask_index_selected_bw = index_iwe_bw_padded > -1
        index_iwe_bw_padded[torch.logical_not(mask_index_selected_bw)] = 0

        # make search window for each eci pixel
        N = self.config['loss']['edge']['search_window']
        # get the serach window of each vert
        x, y = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='xy')
        xys = (torch.cat([y[None], x[None]], dim=0)).expand(num_pairs, N_selected, 2, N, N).type_as(iwe_fw) # .to(device)

        locat_fw = xys + index_iwe_fw_padded[..., None, None].expand(num_pairs, N_selected, 2, N, N)
        locat_fw[:, :, 0, :, :] *= (self.width + N - 1)
        idx_fw = torch.sum(locat_fw, dim=2).view(num_pairs, N_selected, N * N)
        idx_new_fw = idx_fw.view(idx_fw.shape[0], idx_fw.shape[1] * idx_fw.shape[2]).long()

        locat_bw = xys + index_iwe_bw_padded[..., None, None].expand(num_pairs, N_selected, 2, N, N)
        locat_bw[:, :, 0, :, :] *= (self.width + N - 1)
        idx_bw = torch.sum(locat_bw, dim=2).view(num_pairs, N_selected, N * N)
        idx_new_bw = idx_bw.view(idx_bw.shape[0], idx_bw.shape[1] * idx_bw.shape[2]).long()
        # use edge image to alleviate the computation cost
        kps_center_tmp_fw = F.pad(kps_center_fw, (0, 0, N // 2 - 1, N // 2, N // 2 - 1, N // 2), "constant", -50)
        kps_center_tmp_fw = kps_center_tmp_fw.reshape(num_pairs, -1, 2)
        kps_center_tmp_x_fw = torch.gather(kps_center_tmp_fw[..., 1], 1, idx_new_fw)
        kps_center_tmp_y_fw = torch.gather(kps_center_tmp_fw[..., 0], 1, idx_new_fw)
        kps_center_tmp_x_fw = kps_center_tmp_x_fw.reshape(num_pairs, N_selected, N * N)
        kps_center_tmp_y_fw = kps_center_tmp_y_fw.reshape(num_pairs, N_selected, N * N)
        center_fw = index_iwe_fw_padded[..., None].expand(num_pairs, N_selected, 2, N * N)
        # compute the distance of each pixel
        dist_fw = torch.sqrt((center_fw[:, :, 0, :] - kps_center_tmp_y_fw) ** 2 + (center_fw[:, :, 1, :] - kps_center_tmp_x_fw) ** 2 + \
                          self.config['loss']['edge']['dist_bias'])

        kps_center_tmp_bw = F.pad(kps_center_bw, (0, 0, N // 2 - 1, N // 2, N // 2 - 1, N // 2), "constant", -50)
        kps_center_tmp_bw = kps_center_tmp_bw.reshape(num_pairs, -1, 2)
        kps_center_tmp_x_bw = torch.gather(kps_center_tmp_bw[..., 1], 1, idx_new_bw)
        kps_center_tmp_y_bw = torch.gather(kps_center_tmp_bw[..., 0], 1, idx_new_bw)
        kps_center_tmp_x_bw = kps_center_tmp_x_bw.reshape(num_pairs, N_selected, N * N)
        kps_center_tmp_y_bw = kps_center_tmp_y_bw.reshape(num_pairs, N_selected, N * N)
        center_bw = index_iwe_bw_padded[..., None].expand(num_pairs, N_selected, 2, N * N)
        dist_bw = torch.sqrt(
            (center_bw[:, :, 0, :] - kps_center_tmp_y_bw) ** 2 + (center_bw[:, :, 1, :] - kps_center_tmp_x_bw) ** 2 + \
            self.config['loss']['edge']['dist_bias'])

        # find the vertices with max weights
        mesh_edge_new_fw = F.pad(mesh_edge_fw, (N // 2 - 1, N // 2, N // 2 - 1, N // 2), "constant", 0)
        mesh_edge_new_fw = mesh_edge_new_fw.view(mesh_edge_new_fw.shape[0], mesh_edge_new_fw.shape[1] * mesh_edge_new_fw.shape[2])
        mesh_weight_all_fw = torch.gather(mesh_edge_new_fw, 1, idx_new_fw)
        mesh_weight_all_fw = mesh_weight_all_fw.view(num_pairs, N_selected, N * N)

        metrics_near_fw = mesh_weight_all_fw / (dist_fw + 1e-6)
        idx_searched_fw = torch.argmax(metrics_near_fw, dim=2, keepdim=True)
        mesh_weight_fw = torch.gather(metrics_near_fw, 2, idx_searched_fw) * (torch.gather(dist_fw, 2, idx_searched_fw))
        mask_mesh_kps_fw = (torch.gather(dist_fw, 2, idx_searched_fw)) < 1000

        gt_mask = torch.ones(batch_size, seq_len, 1).type_as(iwe_fw)
        gt_mask[:, seq_len//2] *= 0.

        # sum up the distance loss
        loss_event_fw = mesh_weight_fw * torch.gather(dist_fw, 2, idx_searched_fw) * mask_index_selected_fw[..., 0:1] * mask_mesh_kps_fw
        loss_numerator_fw = torch.sum(loss_event_fw.reshape(batch_size, seq_len-1, N_selected) * gt_mask[:, 1:])
        weight_divide_fw = (mask_index_selected_fw[..., 0:1] * mask_mesh_kps_fw).float() * mesh_weight_fw
        loss_denominator_fw = torch.sum(weight_divide_fw.reshape(batch_size, seq_len-1, N_selected) * gt_mask[:, 1:])
        loss_fw = loss_numerator_fw / (loss_denominator_fw + 1e-6)

        mesh_edge_new_bw = F.pad(mesh_edge_bw, (N // 2 - 1, N // 2, N // 2 - 1, N // 2), "constant", 0)
        mesh_edge_new_bw = mesh_edge_new_bw.view(mesh_edge_new_bw.shape[0],
                                                 mesh_edge_new_bw.shape[1] * mesh_edge_new_bw.shape[2])
        mesh_weight_all_bw = torch.gather(mesh_edge_new_bw, 1, idx_new_bw)
        mesh_weight_all_bw = mesh_weight_all_bw.view(num_pairs, N_selected, N * N)

        metrics_near_bw = mesh_weight_all_bw / (dist_bw + 1e-6)
        idx_searched_bw = torch.argmax(metrics_near_bw, dim=2, keepdim=True)
        mesh_weight_bw = torch.gather(metrics_near_bw, 2, idx_searched_bw) * (torch.gather(dist_bw, 2, idx_searched_bw))
        mask_mesh_kps_bw = (torch.gather(dist_bw, 2, idx_searched_bw)) < 1000

        loss_event_bw = mesh_weight_bw * torch.gather(dist_bw, 2, idx_searched_bw) * mask_index_selected_bw[...,
                                                                                     0:1] * mask_mesh_kps_bw
        loss_numerator_bw = torch.sum(loss_event_bw.reshape(batch_size, seq_len-1, N_selected) * gt_mask[:, :-1])
        weight_divide_bw = (mask_index_selected_bw[..., 0:1] * mask_mesh_kps_bw).float() * mesh_weight_bw
        loss_denominator_bw = torch.sum(weight_divide_bw.reshape(batch_size, seq_len-1, N_selected) * gt_mask[:, :-1])
        loss_bw = loss_numerator_bw / (loss_denominator_bw + 1e-6)

        return loss_fw, loss_bw

    def get_mask_mse_loss(self, a, b, mask):
        if len(mask.shape) != 3:
            mask = mask[..., None]
        error_sum = torch.sum(((a - b)*mask)**2)
        mean = torch.sum(mask)
        return error_sum / (mean + 1e-6)

    def get_dict_clone(self, dicts):
        res = {}
        for key in dicts.keys():
            if dicts[key] is dict:
                res[key] = self.get_dict_clone(dicts[key])
            else:
                res[key] = dicts[key].clone()
        return res

    def get_hand_flow(self, manos):
        # get hand flow
        flow_fw, flow_mask = self.forward_flow(manos)
        if self.config['log']['verbose']:
            print('flow_max: ', flow_fw.max())

        return flow_fw

    def get_supervision(self, manos, batch, flow_fw):
        tmp_idx = self.seq_len - self.annot_len
        # get smooth loss
        smooth_loss = torch.tensor(0.0)
        if self.config['loss']['smooth']['weight'] != 0.0:
            smooth_loss, smooth_loss_origin = self.get_smooth_loss(manos)
            if self.config['log']['verbose']:
                print('smooth loss: ', smooth_loss_origin)

        # get event slice
        event_indices_cm = self.get_event_slice(1., batch['Num_events'][:, tmp_idx+1:])

        # warp events to get image with warped events
        iwe_fw, iwe_bw = self.warp(
            flow_fw,
            [0., 1.],
            batch['events'][:, tmp_idx+1:],
            batch['pol_mask'][:, tmp_idx+1:],
            events_indices=event_indices_cm,
            scale=1.0,
        )

        if self.config['log']['imshow']:
            plt.imshow((flow_to_image(flow_fw[0, -1, ..., 0].detach().cpu().numpy(),
                                      flow_fw[0, -1, ..., 1].detach().cpu().numpy()) * 255).astype(np.uint32))
            plt.show()
            pass

        loss_cm_fw, loss_cm_bw = self.get_cm_loss(iwe_fw, iwe_bw)
        cm_loss = loss_cm_fw + loss_cm_bw
        if self.config['log']['verbose']:
            print('cm_loss: ', cm_loss)

        edge_loss = torch.tensor(0.0)
        if self.config['loss']['edge']['weight'] != 0.0:
            factor = self.config['loss']['edge']['factor']
            # get edge slice for edge loss
            event_indices_edge = self.get_event_slice(factor,
                                                      batch['Num_events'][:, tmp_idx+1:])
            # get iwe for edge loss
            iwe_fw_edge, iwe_bw_edge = self.warp(
                flow_fw,
                [0., 1.],
                batch['events'][:, tmp_idx+1:],
                batch['pol_mask'][:, tmp_idx+1:],
                events_indices=event_indices_edge,
                scale=1.0
            )
            # get edge loss
            loss_fw_edge, loss_bw_edge = self.get_edge_loss(iwe_fw_edge, iwe_bw_edge, manos)
            edge_loss = loss_fw_edge + loss_bw_edge

            if self.config['log']['verbose']:
                print('edge loss: ', edge_loss)

        loss_supervision = smooth_loss * self.config['loss']['smooth']['weight'] + \
                    cm_loss * self.config['loss']['cm']['weight'] + \
                    edge_loss * self.config['loss']['edge']['weight']
        return loss_supervision, smooth_loss, cm_loss, edge_loss, iwe_fw, iwe_bw

    def get_mano_loss(self, manos_buffer, preds, gt_mano_mask):
        # mano loss
        trans_loss = self.get_mask_mse_loss(manos_buffer['mano_trans'],
                                preds['mano_trans'], gt_mano_mask)
        rot_loss = self.get_mask_mse_loss(manos_buffer['mano_rot_pose'],
                              preds['mano_rot_pose'], gt_mano_mask)
        if self.config['exper']['exper_name'] == 'train_evhands_eventhand' and self.config['exper']['mode'] == 'eval':
            hand_pose_loss = 0.
        else:
            hand_pose_loss = self.get_mask_mse_loss(manos_buffer['mano_hand_pose'],
                                        preds['mano_hand_pose'], gt_mano_mask)
        shape_loss = self.get_mask_mse_loss(manos_buffer['mano_shape'],
                                preds['mano_shape'], gt_mano_mask)
        mano_loss = self.config['loss']['mano']['trans_weight'] * trans_loss \
                    + self.config['loss']['mano']['rot_weight'] * rot_loss \
                    + self.config['loss']['mano']['hand_pose_weight'] * hand_pose_loss \
                    + self.config['loss']['mano']['shape_weight'] * shape_loss
        return mano_loss, trans_loss, rot_loss, hand_pose_loss, shape_loss

    def get_joint3d_loss(self, manos_buffer, preds, gt_mano_mask):
        if self.config['exper']['exper_name'] == 'train_evhands_eventhand' and self.config['exper']['mode'] == 'eval':
            output_pred = self.mano_layer(
                global_orient=preds['mano_rot_pose'].reshape(-1, 3),
                hand_pose=preds['mano_hand_pose'].reshape(-1, 6),
                betas=torch.zeros((*preds['mano_hand_pose'].shape[:-1],10),device=preds['mano_rot_pose'].device).reshape(-1, 10),
                transl=preds['mano_trans'].reshape(-1, 3)
            )
            mano_layer_gt = MANO(self.config['data']['smplx_path'], use_pca=True, is_rhand=True, num_pca_comps=45).to(preds['mano_rot_pose'].device)
            output_manos_buffer = mano_layer_gt(
                global_orient=manos_buffer['mano_rot_pose'].reshape(-1, 3),
                hand_pose=manos_buffer['mano_hand_pose'].reshape(-1, 45),
                betas=manos_buffer['mano_shape'].reshape(-1, 10),
                transl=manos_buffer['mano_trans'].reshape(-1, 3)
            )
        elif self.config['exper']['exper_name'] == 'train_evhands_eventhand' and self.config['exper']['mode'] != 'eval':
            output_pred = self.mano_layer(
                global_orient=preds['mano_rot_pose'].reshape(-1, 3),
                hand_pose=preds['mano_hand_pose'].reshape(-1, 6),
                betas=torch.zeros((*preds['mano_hand_pose'].shape[:-1], 10),
                                  device=preds['mano_rot_pose'].device).reshape(-1, 10),
                transl=preds['mano_trans'].reshape(-1, 3)
            )
            output_manos_buffer = self.mano_layer(
                global_orient=manos_buffer['mano_rot_pose'].reshape(-1, 3),
                hand_pose=manos_buffer['mano_hand_pose'].reshape(-1, 6),
                betas=manos_buffer['mano_shape'].reshape(-1, 10),
                transl=manos_buffer['mano_trans'].reshape(-1, 3)
            )
        else:
            output_pred = self.mano_layer(
                global_orient=preds['mano_rot_pose'].reshape(-1, 3),
                hand_pose=preds['mano_hand_pose'].reshape(-1, 45),
                betas=preds['mano_shape'].reshape(-1, 10),
                transl=preds['mano_trans'].reshape(-1, 3)
            )
            output_manos_buffer = self.mano_layer(
                global_orient=manos_buffer['mano_rot_pose'].reshape(-1, 3),
                hand_pose=manos_buffer['mano_hand_pose'].reshape(-1, 45),
                betas=manos_buffer['mano_shape'].reshape(-1, 10),
                transl=manos_buffer['mano_trans'].reshape(-1, 3)
            )
        joints_pred = output_pred.joints.reshape(preds['mano_rot_pose'].shape[0], self.annot_len, 21, 3)
        joints_manos_buffer = output_manos_buffer.joints.reshape(preds['mano_rot_pose'].shape[0], self.annot_len, 21, 3)

        joints_pred_align = joints_manos_buffer[:, :, :1, :] - joints_pred[:, :, :1, :] + joints_pred
        error_sum = torch.sum(((joints_pred_align - joints_manos_buffer) * gt_mano_mask[..., None])**2)
        mask_sum = torch.sum(gt_mano_mask)
        joints_3d_mse = error_sum / (mask_sum + 1e-6) / 20.
        return joints_3d_mse, joints_pred

    def get_flow_loss(self, flow_pred, flow_fw, batch):
        flow_loss = torch.tensor(0.).type_as(batch['annot']['mano_trans'])
        if flow_pred is None:
            pesudo_gt_flow = None
            pass
        elif self.config['preprocess']['bbox']['usage']:
            bbox_size = self.config['preprocess']['bbox']['size']
            pesudo_gt_flow = torch.zeros((flow_fw.shape[0], flow_fw.shape[1], bbox_size, bbox_size, 2), dtype=torch.float32).type_as(batch['annot']['mano_trans'])
            for i in range(flow_fw.shape[0]):
                for j in range(flow_fw.shape[1]):
                    bbox_info = batch['bbox'][i, -self.annot_len + j + 1].clone()
                    if bbox_info[0] <= 0.3:
                        continue
                    # make the flow_fw match the scale and trans augmentation of bbox
                    flow_fw_tmp = torch.nn.functional.interpolate(
                        flow_fw[i, j:j+1].permute(0, 3, 1, 2).detach() * bbox_info[0],
                        size=(int(bbox_info[0]*self.height), int(bbox_info[0]*self.width)),
                        mode='bilinear',
                        align_corners=True
                    )
                    c_xy = batch['annot']['K'][i, -self.annot_len + j+1, :2, 2]
                    bbox_info[1:] -= (1-bbox_info[0])*c_xy
                    xy = bbox_info[1:].int()
                    top = max(xy[1], 0)
                    pad_top = top - xy[1]
                    down = min(xy[1]+bbox_size, int(bbox_info[0]*self.height))
                    pad_down = xy[1]+bbox_size - down

                    left = max(xy[0], 0)
                    pad_left = left - xy[0]
                    right = min(xy[0] + bbox_size, int(bbox_info[0]*self.width))
                    pad_right = xy[0] + bbox_size - right
                    # crop the flow
                    flow_fw_tmp_crop = flow_fw_tmp[:, :, top:down, left:right]
                    # padding flow
                    flow_fw_tmp_crop = torch.nn.functional.pad(flow_fw_tmp_crop, (pad_left, pad_right, pad_top, pad_down)).permute(0, 2, 3, 1)[0]
                    flow_fw_tmp_crop = flow_fw_tmp_crop[:bbox_size, :bbox_size, :]
                    flow_fw_tmp_crop = flow_fw_tmp_crop.repeat(1, 1, self.config['method']['flow']['num_encoders'])
                    # EPE loss
                    flow_loss += torch.sum(
                        (flow_fw_tmp_crop / bbox_size * 8. - flow_pred[i, -self.annot_len + j + 1]) ** 2
                     ) / (flow_fw.shape[0] * flow_fw.shape[1] * bbox_size * bbox_size * flow_fw_tmp_crop.shape[2])

                    pesudo_gt_flow[i, j] = flow_fw_tmp_crop[:, :, -2:] # / bbox_size * 8.
        else:
            pesudo_gt_flow = torch.zeros((flow_fw.shape[0], flow_fw.shape[1], self.config['data']['height'], self.config['data']['width'], 2),
                                         dtype=torch.float32).type_as(batch['annot']['mano_trans'])
            for i in range(flow_fw.shape[0]):
                for j in range(flow_fw.shape[1]):
                    # make the flow_fw match the scale and trans augmentation of bbox
                    flow_fw_tmp = torch.nn.functional.interpolate(
                        flow_fw[i, j:j+1].permute(0, 3, 1, 2).detach(),
                        size=(self.height, self.width),
                        mode='bilinear',
                        align_corners=True
                    )
                    flow_fw_tmp = flow_fw_tmp[0].permute(1,2,0)
                    flow_fw_tmp = flow_fw_tmp.repeat(1, 1, self.config['method']['flow']['num_encoders'])
                    # EPE loss
                    flow_loss += torch.sum(
                        (flow_fw_tmp  * 8. - flow_pred[i, -self.annot_len + j + 1]) ** 2
                     ) / (flow_fw.shape[0] * flow_fw.shape[1]  * flow_fw_tmp.shape[2])
                    pesudo_gt_flow[i, j] = flow_fw_tmp[:, :, -2:]

        if self.config['log']['verbose']:
            print('flow loss: ', flow_loss)
        return flow_loss, pesudo_gt_flow

    def forward(self, batch, preds, flow_pred, mode='train'):
        batch_size = batch['annot']['mano_trans'].shape[0]
        tmp_idx = self.seq_len - self.annot_len

        gt_mano_mask = (batch['ids'][:, tmp_idx:] != -1)[..., None]

        self.R = batch['annot']['R'][:, tmp_idx:]
        self.t = batch['annot']['t'][:, tmp_idx:]
        self.K = batch['annot']['K'][:, tmp_idx:]

        loss_all = torch.tensor(0.).type_as(self.R)
        edge_loss = torch.tensor(0.).type_as(self.R)
        cm_loss = torch.tensor(0.).type_as(self.R)
        smooth_loss = torch.tensor(0.).type_as(self.R)

        iwe_fw = torch.zeros((batch_size, self.annot_len-1, self.height, self.width, 2), dtype=torch.float32).type_as(batch['annot']['mano_trans'])
        iwe_bw = torch.zeros_like(iwe_fw).type_as(batch['annot']['mano_trans'])

        manos_buffer = {}
        # for fitting
        for key in preds.keys():
            manos_buffer[key] = batch['annot'][key][:, tmp_idx:]

        manos = {}
        # if there is gt mano results, use it for supervision
        # TODO
        for key in preds.keys():
            manos[key] = preds[key].clone() * torch.logical_not(gt_mano_mask) + batch['annot'][key][:, tmp_idx:] * gt_mano_mask

        flow_fw = self.get_hand_flow(manos)
        if 'semi-supervision' in self.config['exper']['supervision']:
            loss_supervision, smooth_loss, cm_loss, edge_loss, iwe_fw, iwe_bw = self.get_supervision(manos, batch, flow_fw)
            loss_all += loss_supervision * self.config['loss']['supervision']['weight']

        if mode == 'val' or mode == 'eval':
            for key in manos_buffer.keys():
                manos_buffer[key] = batch['annot'][key][:, tmp_idx:]

        flow_loss = torch.tensor(0.0)
        if self.config['loss']['flow']['weight'] != 0.0:
            flow_loss, pesudo_gt_flow = self.get_flow_loss(flow_pred, flow_fw, batch)

        if self.config['method']['supervision_type'] == 'mano_loss':
            gt_mano_mask = torch.ones_like(gt_mano_mask).type_as(gt_mano_mask)
        # mano loss
        mano_loss, trans_loss, rot_loss, hand_pose_loss, shape_loss = self.get_mano_loss(manos_buffer, preds, gt_mano_mask)
        joints_3d_loss, joints_pred = self.get_joint3d_loss(manos_buffer, preds, gt_mano_mask)

        if self.config['log']['verbose']:
            print('mano_loss: ', mano_loss, trans_loss, rot_loss, hand_pose_loss, shape_loss)
            print('joints_3d_loss: ', joints_3d_loss)

        loss_all += mano_loss * self.config['loss']['mano']['weight'] + \
                    flow_loss * self.config['loss']['flow']['weight'] + \
                    joints_3d_loss * self.config['loss']['joint_3d']['weight']

        iwe_fw = iwe_fw.reshape(batch_size, self.annot_len - 1, iwe_fw.shape[1], iwe_fw.shape[2], iwe_fw.shape[3])
        iwe_bw = iwe_bw.reshape(batch_size, self.annot_len - 1, iwe_bw.shape[1], iwe_bw.shape[2], iwe_bw.shape[3])
        # TODO
        res = {
            'iwe_bw': iwe_bw,
            'joints': joints_pred,
            'all_loss': loss_all,
            'mano_loss': mano_loss,
            'flow_loss': flow_loss,
            'smooth_loss': smooth_loss,
            'edge_loss': edge_loss,
            'cm_loss': cm_loss,
            'joint_loss': joints_3d_loss,
            'trans_loss': trans_loss,
            'rot_loss': rot_loss,
            'hand_pose_loss': hand_pose_loss,
            'shape_loss': shape_loss,
        }

        if self.config['log']['gt_show']:
            res['pesudo_gt_flow'] = pesudo_gt_flow

        if flow_pred is not None:
            res['flow_pred'] = flow_pred

        if mode == 'eval' or self.config['log']['imshow']:
            eci = 1 * batch['eci'][:, -self.annot_len:, -2:].permute(0, 1, 3, 4, 2)
            render_preds = {}
            for key in preds.keys():
                render_preds[key] = preds[key][:, -self.annot_len:].clone()
            if 'aligned_mesh' in self.config['log'].keys():
                # TODO the fast motion sequences
                if self.config['log']['aligned_mesh']:
                    valid = batch['annot']['fast'][:, -1] == 0
                    render_preds['mano_trans'][valid] = batch['annot']['mano_trans'][valid, -self.annot_len:].clone()
            if self.config['log']['render_background']:
                bg = eci
            else:
                bg = torch.zeros_like(eci).type_as(eci)
            render_imgs = self.render_mesh(render_preds, bg)
            output_pred = self.mano_layer(
                global_orient=preds['mano_rot_pose'].reshape(-1, 3),
                hand_pose=preds['mano_hand_pose'].reshape(-1, 45),
                betas=preds['mano_shape'].reshape(-1, 10),
                transl=preds['mano_trans'].reshape(-1, 3)
            )
            if self.config['log']['show_mesh']:
                verts = output_pred.vertices.reshape(batch_size, self.annot_len, -1, 3)
                res['verts'] = verts
                res['faces'] = torch.tensor(self.mano_layer.faces.astype(np.int32), dtype=torch.int32).type_as(self.R)

        # for visualization
        if mode == 'eval' and self.config['log']['save_crop']:
            bbox_size = self.config['preprocess']['bbox']['size']
            iwe_fw_tmp = torch.zeros(batch_size, self.annot_len-1, bbox_size, bbox_size, iwe_fw.shape[4]).type_as(self.R)
            flow_fw_tmp = torch.zeros(batch_size, self.annot_len-1, bbox_size, bbox_size, flow_fw.shape[4]).type_as(self.R)
            render_imgs_tmp = torch.zeros(batch_size, self.annot_len, bbox_size, bbox_size, 3).type_as(self.R)
            for i in range(batch_size):
                for j in range(self.annot_len - 1):
                    bbox_info = batch['bbox'][i, -self.annot_len + j + 1]
                    tl = bbox_info[1:]
                    xy_tmp = tl + bbox_size / 2
                    xy = (xy_tmp - self.K[-1, 0, :2, 2])/bbox_info[0] + self.K[-1, 0, :2, 2]
                    center = (xy - bbox_size / 2).int()
                    if center[1] < 0 or center[1] + bbox_size >= self.height or center[0] < 0 or center[1] + bbox_size >= self.width:
                        continue
                    iwe_fw_tmp[i, j] = iwe_fw[i, j, center[1]:center[1] + bbox_size,
                                       center[0]:center[0] + bbox_size]
                    flow_fw_tmp[i, j] = flow_fw[i, j, center[1]:center[1] + bbox_size,
                                       center[0]:center[0] + bbox_size]
                    render_imgs_tmp[i, j+1] = render_imgs[i, j+1, center[1]:center[1] + bbox_size,
                                       center[0]:center[0] + bbox_size]
            iwe_fw = iwe_fw_tmp
            flow_fw = flow_fw_tmp
            render_imgs = render_imgs_tmp
        if mode == 'eval' and self.config['exper']['exper_name'] != 'train_evhands_eventhand':
            res['iwe_fw'] = iwe_fw
            res['flow_fw'] = flow_fw
            res['render_imgs'] = render_imgs

        return res

def compute_normals(verts, faces):
    device = verts.device
    faces = torch.tensor(faces.astype(np.int32)).to(device)
    verts_normals = torch.zeros_like(verts).to(device)
    vertices_faces = torch.stack([verts[i][faces.long()] for i in range(verts.shape[0])])
    verts_normals = verts_normals.index_add(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals = verts_normals.index_add(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals = verts_normals.index_add(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=2)
    return verts_normals