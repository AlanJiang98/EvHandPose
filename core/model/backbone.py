import torch
import torch.nn as nn
import numpy as np
from core.model.utils.unet import MultiResUNet
from core.model.utils.basic_models import BasicBlock, Bottleneck, conv1x1, RecurrentLayer
from core.model.utils.spike_tensor_encoder import QuantizationLayer
from core.model.utils.lstm.net_matrixlstm import MatrixLSTMResNet
from fvcore.nn import FlopCountAnalysis
import time

class FlowNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        FlowNet_kwargs = {
            'base_num_channels': self.config['method']['flow']['base_num_channels'],
            'num_encoders': self.config['method']['flow']['num_encoders'],
            'num_residual_blocks': self.config['method']['flow']['num_residual_blocks'],
            'num_output_channels': 2,
            'skip_type': 'concat',
            'norm': None,
            'num_bins': self.config['method']['flow']['num_bins'],
            'use_upsample_conv': True,
            'kernel_size': self.config['method']['flow']['kernel_size'],
            'channel_multiplier': 2,
            'final_activation': 'tanh',
        }
        self.num_encoders = FlowNet_kwargs['num_encoders']
        self.multires_unet = MultiResUNet(FlowNet_kwargs)

    def reset_states(self):
        pass

    def forward(self, inp_voxel):
        '''
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        '''

        # pad input
        x = inp_voxel

        # forward pass
        multires_flow = self.multires_unet.forward(x)

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            scale1 = multires_flow[-1].shape[2] / flow.shape[2]
            scale2 = multires_flow[-1].shape[3] / flow.shape[3]
            if torch.is_tensor(scale1):
                scale1 = scale1.item()
                scale2 = scale2.item()
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        scale1,
                        scale2,
                    ),
                )
            )
        return torch.stack(flow_list, dim=1)

    def __str__(self):
        '''
        Model prints with number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Backbone(nn.Module):
    def __init__(self, config):
        super(Backbone, self).__init__()
        self.config = config
        self.base_channels = self.config['method']['event_encoder']['base_num_channels']
        self.dilation = 1
        if self.config['method']['block'] == 'BasicBlock':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.inplanes = self.base_channels

        self.event_encoders = self.get_event_encoders()
        self.fusion_encoders = self.get_fusion_layers()
        self.sequence_encoders = self.get_sequence_layers()
        self.predictors = self.get_predictor_layers()
        self.init_weights()

    def get_event_encoders(self):
        layers = nn.ModuleList()
        if self.config['preprocess']['repre'] == 'LNES':
            factor = 2
        else:
            factor = 1
        if len(self.config['method']['event_encoder']['channel_sizes']) > 0:
            layers.append(
                nn.Conv2d(
                    self.config['method']['event_encoder']['num_bins'] * factor,
                    self.base_channels,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(self.base_channels))
            layers.append(nn.ReLU(inplace=True))
            for i, num_layers in enumerate(self.config['method']['event_encoder']['channel_sizes']):
                layers.append(
                    self._make_layer(
                        self.block, self.block.expansion**i * self.base_channels, num_layers
                    )
                )
        return layers

    def get_fusion_layers(self):
        layers = nn.ModuleList()
        factor = 2 if self.config['preprocess']['repre'] == 'LNES' else 1
        tmp_inplanes = 0
        if self.config['method']['event_encoder']['usage']:
            if len(self.config['method']['event_encoder']['channel_sizes']) < 1:
                tmp_inplanes += self.config['method']['event_encoder']['num_bins'] * factor
            else:
                tmp_inplanes += self.inplanes

        if self.config['method']['flow']['usage']:
            if 'last_for_output' in self.config['method']['flow'].keys() and self.config['method']['flow']['last_for_output']:
                tmp_inplanes += 2
            else:
                tmp_inplanes += 2 * self.config['method']['flow']['num_encoders']
        layers.append(
            nn.Conv2d(
                tmp_inplanes,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
            )
        )
        layers.append(nn.BatchNorm2d(self.inplanes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        #change here!
        inplane = self.inplanes
        for i, num_layers in enumerate(self.config['method']['encoder_fusion']['channel_sizes']):
            layers.append(
                self._make_layer( # 2**i * self.base_channels # self.block.expansion**i * self.inplanes
                    self.block, 2**i * inplane, num_layers, stride=2
                )
            )
        return layers

    def get_sequence_layers(self):
        layers = nn.ModuleList()
        for i in range(self.config['method']['seq_model']['num_layers']):
            layers.append(
                RecurrentLayer(
                    self.config['method']['seq_model']['model'],
                    self.inplanes,
                    self.inplanes,
                    kernel_size=3
                )
            )
            layers.append(
                nn.BatchNorm3d(num_features=self.inplanes)
            )
        return layers

    def get_predictor_layers(self):
        layers = nn.ModuleList()
        layers.append(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        layer_dims = self.config['method']['predictor']['layer_dims']
        layers.append(nn.Linear(self.inplanes, layer_dims[0]))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        for i in range(len(layer_dims)-1):
            layers.append(
                nn.Linear(layer_dims[i], layer_dims[i+1])
            )
            if i != len(layer_dims)-2:
                layers.append(nn.LeakyReLU(0.1, inplace=True))
        return layers

    def forward(self, event_repre, flow=None, num_events=None):
        if 'last_for_output' in self.config['method']['flow'].keys() and self.config['method']['flow']['last_for_output']:
            flow = flow[:, :, -2:].clone()

        shape1 = event_repre.shape
        x = event_repre.view(-1,*shape1[2:])

        for encoder in self.event_encoders:
            x = encoder(x)

        shape2 = x.shape
        x = x.view(shape1[0], shape1[1], *shape2[1:])
        if self.config['method']['flow']['usage'] and self.config['method']['event_encoder']['usage']:
            x = torch.cat([x, flow], dim=2)
        elif self.config['method']['flow']['usage'] and not self.config['method']['event_encoder']['usage']:
            x = flow
        elif not self.config['method']['flow']['usage'] and self.config['method']['event_encoder']['usage']:
            x = x
        else:
            raise ValueError('No info for predicting branch! ')
        x = x.view(shape1[0] * shape1[1], -1, shape2[2], shape2[3])
        for encoder in self.fusion_encoders:
            x = encoder(x)

        shape3 = x.shape
        x = x.view(shape1[0], shape1[1], shape3[1], shape3[2], shape3[3])
        x = x.permute(0, 2, 1, 3, 4)

        for encoder in self.sequence_encoders:
            x = encoder(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x[:, -self.config['method']['seq_model']['annot_len']:].clone()
        shape4 = x.shape
        x = x.reshape(-1, shape4[2], shape4[3], shape4[4])

        for i, predictor in enumerate(self.predictors):
            x = predictor(x)
            if i == 0:
                x = x.view(x.shape[0], -1)

        x = x.view(shape4[0], shape4[1], -1)
        mean = torch.tensor(self.config['method']['predictor']['mean'], dtype=torch.float32).type_as(x)
        scale = torch.tensor(self.config['method']['predictor']['scale'], dtype=torch.float32).type_as(x)
        output = {}
        output['mano_hand_pose'] = x[:, :, :45].clone()
        output['mano_trans_tmp'] = x[:, :, 45:48].clone() * scale[None, None, :] + mean[None, None, :]
        output['mano_shape'] = x[:, :, 48:58].clone()
        output['mano_rot_pose'] = x[:, :, 58:].clone()
        return output

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 1,
                            self.base_channels, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=1,
                                base_width=self.base_channels, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)