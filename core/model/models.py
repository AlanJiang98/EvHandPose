import torch
from torch.optim import *
import pytorch_lightning as pl
from abc import abstractmethod
import gc
from core.model.backbone import FlowNet, Backbone
from core.model.supervision import SupervisionBranch
import multiprocessing as mp
from core.model.smplx.body_models import MANO
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.utils import cameras_from_opencv_projection


class EvHands(pl.LightningModule):
    def __init__(self, config):
        super(EvHands, self).__init__()
        self.config = config
        self.flow_encoder = FlowNet(self.config)
        if config['method']['flow']['fixed']:
            for p in self.flow_encoder.parameters():
                p.requires_grad = False
        print('load flow encoder over!')
        self.infer_encoder = Backbone(self.config)
        print('load infer encoder over!')
        self.supervision = SupervisionBranch(self.config)
        print('load supervision over!')
        self._dataset = None
        self.epoch = 0
        self.automatic_optimization = False
        self.validation_step_outputs = None
        self.test_step_outputs = None
        self.val_datasets = []

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    def forward(self, batch):
        annot_len = self.config['method']['seq_model']['annot_len']
        flows = None

        if self.config['method']['flow']['usage']:
            shape_flow = batch['flow_repre'].shape
            flow_input = batch['flow_repre'].reshape(-1, shape_flow[2], shape_flow[3], shape_flow[4])
            flows = self.flow_encoder(flow_input)
            flows = flows.reshape(shape_flow[0], shape_flow[1], -1, shape_flow[3], shape_flow[4])

        preds = self.infer_encoder(batch['event_encoder_repre'], flows.clone() if flows is not None else None)

        if self.config['method']['flow']['usage']:
            flow_pred = flows[:, -annot_len+1:, :, :, :].clone().permute(0, 1, 3, 4, 2)
        else:
            flow_pred = None
        if self.config['preprocess']['bbox']['usage']:
            trans = preds['mano_trans_tmp'].clone()
            x = trans[..., 0] + trans[..., 2] * batch['bbox'][:, -annot_len:, 1].clone() / batch['annot']['K'][:, -annot_len:, 0, 0].clone()
            y = trans[..., 1] + trans[..., 2] * batch['bbox'][:, -annot_len:, 2].clone() / batch['annot']['K'][:, -annot_len:, 1, 1].clone()
            z = trans[..., 2] * batch['bbox'][:, -annot_len:, 0].clone()
            preds['mano_trans'] = torch.stack([x, y, z], dim=-1)

        else:
            preds['mano_trans'] = preds['mano_trans_tmp'].clone()
        preds.pop('mano_trans_tmp')

        if self.config['log']['gt_show']:
            for key in list(preds.keys()):
                preds[key] = batch['annot'][key][:, -annot_len:]

        res = self.supervision(batch, preds, flow_pred, mode=self._dataset[batch['dataset_index'][0]].mode)

        res['flow_pred'] = flow_pred
        res['preds'] = preds

        return res

    def eval_forward(self, batch):
        annot_len = self.config['method']['seq_model']['annot_len']

        shape_flow = batch['flow_repre'].shape
        flow_input = batch['flow_repre'].reshape(-1, shape_flow[2], shape_flow[3], shape_flow[4])
        flows = self.flow_encoder(flow_input)
        flows = flows.reshape(shape_flow[0], shape_flow[1], -1, shape_flow[3], shape_flow[4])
        preds = self.infer_encoder(batch['event_encoder_repre'], flows.clone() if flows is not None else None)

        if self.config['preprocess']['bbox']['usage']:
            trans = preds['mano_trans_tmp'].clone()
            x = trans[..., 0] + trans[..., 2] * batch['bbox'][:, -annot_len:, 1].clone() / batch['annot']['K'][:, -annot_len:, 0, 0].clone()
            y = trans[..., 1] + trans[..., 2] * batch['bbox'][:, -annot_len:, 2].clone() / batch['annot']['K'][:, -annot_len:, 1, 1].clone()
            z = trans[..., 2] * batch['bbox'][:, -annot_len:, 0].clone()
            preds['mano_trans'] = torch.stack([x, y, z], dim=-1)
        else:
            preds['mano_trans'] = preds['mano_trans_tmp'].clone()
        preds.pop('mano_trans_tmp')

        if self.config['method']['flow']['usage']:
            flow_pred = flows[:, -annot_len+1:, :, :, :].clone().permute(0, 1, 3, 4, 2)
        else:
            flow_pred = None

        return preds,flow_pred

    @abstractmethod
    def update_speed_mat(self, dataset):
        dataset.regenerate_items()

    def configure_optimizers(self):
        optimizers = []

        for key, values in self.config['method']['optimizers'].items():
            if hasattr(self, key):
                optimizers.append(
                    eval(values['name'])(eval('self.' + key + '.parameters()'), lr=values['lr'], weight_decay=values['weight_decay']))

        return optimizers

    def zero_grad_(self):
        if type(self.optimizers()) is list:
            for opt in self.optimizers():
                opt.zero_grad()
        else:
            self.optimizers().zero_grad()

    def step_(self):
        if type(self.optimizers()) is list:
            for opt in self.optimizers():
                opt.step()
        else:
            self.optimizers().step()

    def training_step(self, batch, batch_nb):
        res = self(batch)
        self.zero_grad_()

        res['all_loss'].backward() # for ddp may work multi-GPUS
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        self.step_()
        joint_prog_bar = False
        flow_prog_bar = False
        if self.config['method']['flow']['train']:
            flow_prog_bar = True
        else:
            joint_prog_bar = True
        self.log('train_loss', res['all_loss'], prog_bar=True)
        self.log('joint_loss', res['joint_loss'], prog_bar=joint_prog_bar)
        self.log('train_hand_pose_loss', res['hand_pose_loss'])
        self.log('train_trans_loss', res['trans_loss'])
        self.log('train_shape_loss', res['shape_loss'])
        self.log('train_rot_loss', res['rot_loss'])
        self.log('train_mano_loss', res['mano_loss'])
        self.log('train_flow_loss', res['flow_loss'], prog_bar=flow_prog_bar)
        self.log('train_smooth_loss', res['smooth_loss'])
        self.log('train_edge_loss', res['edge_loss'])
        self.log('train_cm_loss', res['cm_loss'])

    def on_train_epoch_end(self):
        # regenerate the items for semi-supervision
        if (self.epoch + 1) % self.config['preprocess']['update_period'] == 0:
            pool = mp.Pool(8)
            for i in range(len(self._dataset)):
                pool.apply_async(self._dataset[i].regenerate_items, args=())
            pool.close()
            pool.join()
        self.epoch += 1
        gc.collect()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        res = self(batch)
        res_origin = batch
        return {**res_origin, **res}

    def test_step(self, batch, batch_nb):
        res = self(batch)
        pred_joints_align = res['joints'][:, -1] - res['joints'][:, -1, :1, :]
        gt_joints_align = batch['annot']['joint'][:, -1] - batch['annot']['joint'][:, -1, :1, :]
        mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((pred_joints_align - gt_joints_align) ** 2, dim=-1))
        mpjpe = torch.sum(mpjpe_eachjoint_eachitem * batch['annot']['joint_valid'][:, -1, :, 0]) / torch.sum(
            batch['annot']['joint_valid'][:, -1]) * 21. / 20.
        self.test_step_outputs = {
            'test_loss': res['all_loss'],
            'test_trans_loss': res['trans_loss'],
            'test_shape_loss': res['shape_loss'],
            'test_rot_loss': res['rot_loss'],
            'test_hand_pose_loss': res['hand_pose_loss'],
            'test_mano_loss': res['mano_loss'],
            'test_flow_loss': res['flow_loss'],
            'test_smooth_loss': res['smooth_loss'],
            'test_edge_loss': res['edge_loss'],
            'test_cm_loss': res['cm_loss'],
            'test_mpjpe': mpjpe,
        }

        return {
            'test_loss': res['all_loss'],
            'test_trans_loss': res['trans_loss'],
            'test_shape_loss': res['shape_loss'],
            'test_rot_loss': res['rot_loss'],
            'test_hand_pose_loss': res['hand_pose_loss'],
            'test_mano_loss': res['mano_loss'],
            'test_flow_loss': res['flow_loss'],
            'test_smooth_loss': res['smooth_loss'],
            'test_edge_loss': res['edge_loss'],
            'test_cm_loss': res['cm_loss'],
            'test_mpjpe': mpjpe,
        }

    def validation_step(self, batch, batch_nb):
        res = self(batch)
        pred_joints_align = res['joints'][:, -1] - res['joints'][:, -1, :1, :]
        gt_joints_align = batch['annot']['joint'][:, -1] - batch['annot']['joint'][:, -1, :1, :]
        mpjpe_eachjoint_eachitem = torch.sqrt(torch.sum((pred_joints_align - gt_joints_align) ** 2, dim=-1))
        mpjpe = torch.sum(mpjpe_eachjoint_eachitem * batch['annot']['joint_valid'][:, -1, :, 0]) / torch.sum(batch['annot']['joint_valid'][:, -1]) * 21. / 20.
        self.validation_step_outputs = {
            'val_loss': res['all_loss'],
            'val_trans_loss': res['trans_loss'],
            'val_shape_loss': res['shape_loss'],
            'val_rot_loss': res['rot_loss'],
            'val_hand_pose_loss': res['hand_pose_loss'],
            'val_mano_loss': res['mano_loss'],
            'val_flow_loss': res['flow_loss'],
            'val_smooth_loss': res['smooth_loss'],
            'val_edge_loss': res['edge_loss'],
            'val_cm_loss': res['cm_loss'],
            'val_mpjpe': mpjpe,
        }

        return {
            'val_loss': res['all_loss'],
            'val_trans_loss': res['trans_loss'],
            'val_shape_loss': res['shape_loss'],
            'val_rot_loss': res['rot_loss'],
            'val_hand_pose_loss': res['hand_pose_loss'],
            'val_mano_loss': res['mano_loss'],
            'val_flow_loss': res['flow_loss'],
            'val_smooth_loss': res['smooth_loss'],
            'val_edge_loss': res['edge_loss'],
            'val_cm_loss': res['cm_loss'],
            'val_mpjpe': mpjpe,
        }

    def on_validation_epoch_end(self):
        val_res = None
        val_name = 'val_flow_loss' if self.config['method']['flow']['train'] else 'val_mpjpe'
        for key,values in self.validation_step_outputs.items():
            prog_bar = False
            loss = torch.stack([values]).mean()
            if key == val_name:
                prog_bar = True
                val_res = loss
            self.log(key, loss, prog_bar=prog_bar)
        gc.collect()
        return val_res

    def on_test_epoch_end(self):
        test_res = None
        test_name = 'test_flow_loss' if self.config['method']['flow']['train'] else 'test_mpjpe'
        for key, values in self.test_step_outputs.items():
            prog_bar = False
            loss = torch.stack([values]).mean()
            if key == test_name:
                prog_bar = True
                val_res = loss
            self.log(key, loss, prog_bar=prog_bar)
        gc.collect()
        return val_res

    def get_render(self, hw=[920, 1064]):
        self.raster_settings = RasterizationSettings(
            image_size=(hw[0], hw[1]),
            faces_per_pixel=2,
            perspective_correct=True,
            blur_radius=0.,
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

    def render_hand(self, mano_rot_pose, mano_hand_pose, shape, trans, K, R, t, hw, img_bg=None):
        self.get_render(hw)
        if self.config['data']['dataset'] != 'EventHand':
            mano_layer = MANO(self.config['data']['smplx_path'], use_pca=False, is_rhand=True).to(device="cuda:0")
            output = mano_layer(
                global_orient=mano_rot_pose.reshape(-1, 3),
                hand_pose=mano_hand_pose.reshape(-1, 45),
                betas=shape.reshape(-1, 10),
                transl=trans.reshape(-1, 3)
            )
        else:
            mano_layer = MANO(self.config['data']['smplx_path'], use_pca=True, is_rhand=True, num_pca_comps=6).to(device="cuda:0")
            output = mano_layer(
                global_orient=mano_rot_pose.reshape(-1, 3),
                hand_pose=mano_hand_pose.reshape(-1, 6),
                betas=shape.reshape(-1, 10),
                transl=trans.reshape(-1, 3)
            )

        now_vertices = output.vertices
        faces = torch.tensor(mano_layer.faces.astype(np.int32)).repeat(1, 1, 1).type_as(mano_rot_pose)
        verts_rgb = torch.ones_like(mano_layer.v_template).type_as(mano_rot_pose)
        verts_rgb = verts_rgb.expand(1, verts_rgb.shape[0], verts_rgb.shape[1])
        textures = TexturesVertex(verts_rgb)

        mesh = Meshes(
            verts=now_vertices,
            faces=faces,
            textures=textures
        )
        cameras = cameras_from_opencv_projection(
            R=R.reshape(-1, 3, 3).type_as(mano_rot_pose),
            tvec=t.reshape(-1, 3).type_as(mano_rot_pose),
            camera_matrix=K.reshape(-1, 3, 3).type_as(mano_rot_pose),
            image_size=torch.tensor([hw[0], hw[1]]).expand(1, 2).type_as(mano_rot_pose)
        ).to(mano_rot_pose.device)

        self.render.shader.to(mano_rot_pose.device)
        res = self.render(
            mesh,
            cameras=cameras
        )
        img = res[..., :3]
        img = img.reshape(-1, hw[0], hw[1], 3)

        if img_bg is not None:
            mask = res[..., 3:4].reshape(-1, hw[0], hw[1], 1) != 0.
            b_channel = torch.zeros((*img.shape[:-1],1)).type_as(mano_rot_pose)
            img_bg = torch.cat([img_bg.reshape(-1, *img_bg.shape[2:]).permute(0,2,3,1), b_channel], dim=-1).type_as(mano_rot_pose)
            img = torch.clip(img * mask + mask.logical_not() * img_bg, 0, 1)


        return img, img_bg



