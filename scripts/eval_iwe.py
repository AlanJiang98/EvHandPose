import torch
import roma
import numpy as np
import cv2
from core.model.utils.iwe import get_interpolation, interpolate
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

def get_renderer(config):
    raster_settings = RasterizationSettings(
                image_size=(config['data']['height'], config['data']['width']),
                faces_per_pixel=2,
                perspective_correct=True,
                blur_radius=config['loss']['basic']['blur_radius'],
            )
    lights = PointLights(
        location=[[0, 2, 0]],
        diffuse_color=((0.5, 0.5, 0.5),),
        specular_color=((0.5, 0.5, 0.5),)
    )
    render = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftPhongShader(lights=lights)
    )
    return render

def get_event_slice(config, factor, event_nums):
    real_event_num = torch.min(
        event_nums,
        config['preprocess']['num_events'] * torch.ones_like(event_nums)
    )
    event_nums = torch.max(factor * real_event_num, 120 * torch.ones_like(real_event_num))
    event_indices = torch.zeros(4).type_as(event_nums)
    event_indices[0] = real_event_num - event_nums
    event_indices[1] = real_event_num * 1
    event_indices[2] = real_event_num * 0
    event_indices[3] = event_nums
    return event_indices.long()

def warp(config, flow, time_marker, events, events_mask, events_indices, scale=1.0):
    height, width = config['data']['height'], config['data']['width']
    fw_mask = torch.zeros(events_mask.shape).type_as(events)
    bw_mask = torch.zeros(events_mask.shape).type_as(events)
    fw_mask[events_indices[0]:events_indices[1]] = 1
    bw_mask[events_indices[2]:events_indices[3]] = 1

    fw_events_mask = torch.cat([fw_mask * events_mask for i in range(4)], dim=0).unsqueeze(0)
    bw_events_mask = torch.cat([bw_mask * events_mask for i in range(4)], dim=0).unsqueeze(0)

    flow_idx = events[:, :2].long().clone()  # crazy bug if flow_idx is not integer
    flow_idx[:, 1] *= width
    flow_idx = torch.sum(flow_idx, dim=-1)

    flow = flow.view(-1, 2)
    event_flowy = torch.gather(flow[:, 1], 0, flow_idx.long())[:,None]  # vertical component
    event_flowx = torch.gather(flow[:, 0], 0, flow_idx.long())[:,None]  # horizontal component
    event_flow = torch.cat([event_flowx, event_flowy], dim=1)

    events = events.unsqueeze(0)
    event_flow = event_flow.unsqueeze(0)
    fw_idx, fw_weights = get_interpolation(events, event_flow, time_marker[1], (height, width), scale)
    # per-polarity image of (forward) warped events
    fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, (height, width),
                             polarity_mask=fw_events_mask[:, :, 0:1])
    fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, (height, width),
                             polarity_mask=fw_events_mask[:, :, 1:2])

    bw_idx, bw_weights = get_interpolation(events, event_flow, time_marker[0], (height, width), scale)
    # per-polarity image of (forward) warped events
    bw_iwe_pos = interpolate(bw_idx.long(), bw_weights, (height, width),
                             polarity_mask=bw_events_mask[:, :, 0:1])
    bw_iwe_neg = interpolate(bw_idx.long(), bw_weights, (height, width),
                             polarity_mask=bw_events_mask[:, :, 1:2])
    fw_iwe = torch.cat([fw_iwe_pos, fw_iwe_neg], dim=-1)
    bw_iwe = torch.cat([bw_iwe_pos, bw_iwe_neg], dim=-1)
    return fw_iwe, bw_iwe


def get_iwe(config, x, annot, i, device, K_tmp, mano_layer, render):
    '''
    i: batch num
    '''
    pose, rot, shape, trans = annot['mano_hand_pose'][i][-1], annot['mano_rot_pose'][i][-1], annot['mano_shape'][i][-1], \
    annot['mano_trans'][i][-1]
    pose_, rot_, shape_, trans_ = annot['mano_hand_pose'][i][-2], annot['mano_rot_pose'][i][-2], \
        annot['mano_shape'][i][-2], annot['mano_trans'][i][-2]
    coeff_inter = torch.linspace(0., 1., 4).type_as(shape).to(device)
    rot_inter = roma.rotvec_slerp(rot_, rot, coeff_inter).reshape(-1, 3)
    pose_inter = roma.rotvec_slerp(
        pose_.reshape(-1, 3),
        pose.reshape(-1, 3),
        coeff_inter
    ).reshape(-1, 45)
    shape_inter = shape[None, ...].repeat(4, 1) * coeff_inter[None, ...].T + shape_[None, ...].repeat(4, 1) * \
                  (1. - coeff_inter)[None, ...].T
    trans_inter = trans[None, ...].repeat(4, 1) * coeff_inter[None, ...].T + trans_[None, ...].repeat(4, 1) * \
                  (1. - coeff_inter)[None, ...].T
    output = mano_layer(global_orient=rot_inter,
                        hand_pose=pose_inter,
                        betas=shape_inter,
                        transl=trans_inter)
    faces = mano_layer.faces_tensor
    vertices_inter = output.vertices
    kps_inter = torch.bmm(vertices_inter, K_tmp.T[None, ...].expand(4, -1, -1))  # 别忘了转置
    kps_inter = torch.div(kps_inter[:, :, :2], kps_inter[:, :, 2:])

    # for i in range(kps_inter.shape[0]):
    #     img = np.ones((config['data']['height'], config['data']['width'],3))*255.
    #     kps = kps_inter[i].detach().cpu().numpy()
    #     for (x,y) in kps:
    #         cv2.circle(img, (int(x),int(y)), 3, (0,255,0), -1)
    #     cv2.imwrite(f'./{i}_2d.jpg',img)
    #     save_obj(f'./{i}_mesh.obj',vertices_inter[i], faces)

    flow_speed_fw = (kps_inter[-1:] - kps_inter[:-1]) / (1. - coeff_inter[:-1])[None, ...].T[..., None]
    flow_speed_fw = torch.cat([flow_speed_fw, flow_speed_fw[-1:]], dim=0)
    faces = torch.tensor(mano_layer.faces.astype(np.int32))[None, ...].expand(4, -1, -1).type_as(shape)
    cameras = cameras_from_opencv_projection(
        R=torch.eye(3).repeat(4, 1, 1).type_as(shape),
        tvec=torch.zeros(4, 3).type_as(shape),
        camera_matrix=K_tmp[None, ...].expand(4, -1, -1).type_as(shape),
        image_size=torch.tensor([config['data']['height'], config['data']['width']]).expand(4, 2).type_as(shape)
    ).to(device)
    verts_rgb = torch.ones_like(mano_layer.v_template).repeat(4, 1, 1).type_as(shape)
    textures = TexturesVertex(verts_rgb)
    mesh = Meshes(
        verts=vertices_inter.detach(),
        faces=faces,
        textures=textures
    )
    # render.shader.to(shape.device)
    # res = render(mesh, cameras=cameras)
    # for i in range(res.shape[0]):
    #     img = res[i,...,:3]
    #     img = img.detach().cpu().numpy() * 255.
    #     cv2.imwrite(f'./mesh_pro_{i}.jpg', img)
    # assert 0
    frags = render.rasterizer(mesh, cameras=cameras)
    faces_tmp = faces.reshape(-1, 3)
    mask = frags.pix_to_face[..., 0] > -1
    verts_index = faces_tmp[frags.pix_to_face[..., 0]]

    flow_x_fw = torch.gather(flow_speed_fw[..., 0], 1, verts_index.reshape(4, -1).long())
    flow_y_fw = torch.gather(flow_speed_fw[..., 1], 1, verts_index.reshape(4, -1).long())
    flow_x_fw = flow_x_fw.reshape(4, config['data']['height'], config['data']['width'], 3)
    flow_y_fw = flow_y_fw.reshape(4, config['data']['height'], config['data']['width'], 3)
    flow_xy_fw = torch.stack([flow_x_fw, flow_y_fw], dim=-1)
    bary_flow_weights = frags.bary_coords[:, :, :, 0, :].reshape(4, config['data']['height'], config['data']['width'], 3, 1)
    mask = mask.reshape(4, config['data']['height'], config['data']['width'], 1)
    flows_fw = torch.sum(flow_xy_fw * bary_flow_weights * mask[..., None, :], dim=-2)
    weight = mask
    flow_fw = torch.sum(flows_fw, dim=0) / (torch.sum(weight, dim=0) + 1e-7)
    num_events = x['Num_events'][i][-1:]
    event_indices_cm = get_event_slice(config, 1., num_events)
    iwe_fw, _ = warp(
        config,
        flow_fw,
        [0., 1.],
        x['events'][i, -1],
        x['pol_mask'][i, -1],
        events_indices=event_indices_cm,
        scale=1.0,
    )
    return iwe_fw[0]

