import os
import sys
from os.path import dirname
abs_path = os.path.abspath(dirname(dirname(__file__)))
sys.path.insert(0,abs_path)
import numpy as np
import torch
import cv2
import argparse
import copy
import roma
from configs.parser import YAMLParser
import multiprocessing as mp
import resource
import warnings
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
from torch.utils.data import DataLoader
from core.dataset.dataset import EvHandDataset
from core.model.models import EvHands
from pytorch3d.utils import cameras_from_opencv_projection
from core.dataset.dataset_utils import get_dataset_configs
from core.dataset.evaluation_tools import *
from tools.visualization.vis_flow import *
from core.model.smplx.body_models import MANO
from core.dataset.joint_indices import indices_change
from tools.basic_io.json_utils import *
from eval_iwe import *
from tqdm import tqdm


def write_obj(name, vers, tri):
    with open(name, 'w') as file_object:
        for v in vers:
            print("v %f %f %f" % (v[0], v[1], v[2]), file=file_object)
        for f in tri:
            print("f %d %d %d" % (f[0], f[1], f[2]), file=file_object)
def collect_dataset(dataset):
    global datasets
    datasets.append(dataset)

def to_device(x, device, batch=False):
    if type(x) is not dict:
        if x is not None:
            if device == 'cpu':
                if type(x) != float:
                    x = x[None, ...].detach().to(device) if batch else x.detach().to(device)
            else:
                x = x[None, ...].to(device) if batch else x.to(device)
        return x
    else:
        for key in x.keys():
            x[key] = to_device(x[key], device, batch)
        return x


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Evaluation Script')
    parser.add_argument('--config_train', type=str, default='../exper/EvHandOpenSource/train_semi/train.yml')
    parser.add_argument('--config_test', type=str, default='../configs/open_source/eval.yml')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--model_path', type=str, default='../exper/EvHandOpenSource/train_semi/EvHands_epoch=46_val_mpjpe=0.0399.ckpt')
    parser.add_argument('--test_name', type=str, default='test')
    args = parser.parse_args()

    config = YAMLParser(args.config_train)
    config.merge_configs(args.config_test)
    config = config.config
    config['method']['model_path'] = args.model_path

    configs = get_dataset_configs(config, 'eval')
    datasets = []
    print("loading dataset")
    if config['exper']['debug']:
        for config_ in configs:
            # if config_['data']['tmp_cap'] not in ['4','5','26','52']: #'52', '53']:
            #     continue
            print(config_['data']['seq_dir'])
            datasets.append(eval(config['data']['dataset']+'Dataset')(config_))
            if len(datasets) > 3:
                break
    else:
        pool = mp.Pool(8)
        for config_ in configs:
            pool.apply_async(eval(config['data']['dataset']+'Dataset'), args=(config_,), callback=collect_dataset)
        pool.close()
        pool.join()
    test_loaders = []
    print('datasets: ', len(datasets))
    for i, dataset in enumerate(datasets):
        dataset.set_index(i)
        test_loader = DataLoader(dataset, batch_size=config['preprocess']['batch_size'], shuffle=False, num_workers=config['preprocess']['num_workers'],
                   pin_memory=False)
        test_loaders.append(test_loader)
    #torch.cuda.empty_cache()
    device = 'cuda:{}'.format(args.gpu)
    model = eval(config['method']['name']).load_from_checkpoint(checkpoint_path=config['method']['model_path'],
                                                                config=copy.deepcopy(config), map_location=device)
    if config['method']['name'] == 'EvHands':
        model.dataset = datasets
    model.eval()

    root_dir = os.path.join(config['exper']['output_dir'], config['exper']['exper_name'], args.test_name)
    os.makedirs(root_dir, exist_ok=True)
    smplx_path = config['data']['smplx_path']
    if config['exper']['exper_name'] != 'train_evhands_eventhand':
        mano_layer = MANO(smplx_path, use_pca=False, is_rhand=True).to(device)
    else:
        mano_layer = MANO(smplx_path, use_pca=True, is_rhand=True, num_pca_comps=6).to(device)
    faces = mano_layer.faces_tensor
    error_dict = {'normal_fixed':{'mpjpe':[], 'pa-mpjpe':[], '2d-mpjpe':[], '2d-mpjpe_scale':[]},
                  'normal_random':{'mpjpe':[], 'pa-mpjpe':[], '2d-mpjpe':[], '2d-mpjpe_scale':[]},
                  'highlight_fixed':{'mpjpe':[], 'pa-mpjpe':[], '2d-mpjpe':[], '2d-mpjpe_scale':[]},
                  'highlight_random':{'mpjpe':[], 'pa-mpjpe':[], '2d-mpjpe':[], '2d-mpjpe_scale':[]},
                  'flash_fixed':{'mpjpe':[], 'pa-mpjpe':[], '2d-mpjpe':[], '2d-mpjpe_scale':[]},
                  'flash_random':{'mpjpe':[], 'pa-mpjpe':[], '2d-mpjpe':[], '2d-mpjpe_scale':[]},
                  'fast':{'2d-mpjpe':[], '2d-mpjpe_scale':[]},
                  'all_mpjpe':{'mpjpe':[], 'pa-mpjpe':[], '2d-mpjpe':[], '2d-mpjpe_scale':[]}
                  }
    for j, test_loader in enumerate(test_loaders):
        print(50*'*')
        print(datasets[j].config['data']['seq_dir'])
        seq_id = os.path.basename(datasets[j].config['data']['seq_dir'])
        scene = datasets[j].scene
        gesture = datasets[j].gesture_type
        hand_type = datasets[j].hand_type
        camera_annot = json_read(os.path.join(datasets[j].config['data']['data_dir'], 'data', seq_id, 'annot.json'))["camera_info"]
        K = torch.tensor(camera_annot['event']['K'], dtype=torch.float32, device=device)
        mpjpe_seq_error_list = []
        pampjpe_seq_error_list = []
        mpjpe_list = []  # 计算auc曲线存的
        if datasets[j].motion_type == 'fast':
            seq_id = os.path.basename(datasets[j].config['data']['seq_dir'])
            fast_annot_path = os.path.join(config['exper']['fast_annot_path'], seq_id, "annotf.json")
            fast_annot = json_read(fast_annot_path)
            K = torch.tensor(fast_annot['camera_info']['event']['K'], dtype=torch.float32, device=device)
            R = torch.tensor(fast_annot['camera_info']['event']['R'], dtype=torch.float32, device=device)
            t = torch.tensor(fast_annot['camera_info']['event']['T'], dtype=torch.float32, device=device) / 1000.
            hand_type = fast_annot['hand_type']
            fx = K[0,0]
        mpjpe2d_seq_error_list = []
        mpjpe2d_seq_error_list_scale = []
        fast_id = 0  # 用于存fast序列的id
        last_mano = None # 用于算fast序列的iwe
        render = get_renderer(config)
        for x in tqdm(test_loader):
            x = to_device(x, device)
            annot = x['annot']
            if config['method']['name'] == 'EventHands':
                res = model.eval_forward(x)
                output = mano_layer(global_orient=res['mano_rot_pose'],
                                    hand_pose=res['mano_hand_pose'],
                                    betas=res['mano_shape'],
                                    transl=res['mano_trans'])
            else:
                res, flow_pred = model.eval_forward(x)
                if config['exper']['exper_name'] == 'train_evhands_eventhand':
                    res['mano_shape'] = torch.zeros((*res['mano_rot_pose'].shape[:-1], 10), dtype=torch.float32, device=device)
                output = mano_layer(global_orient=res['mano_rot_pose'][:, -1, :],
                                    hand_pose=res['mano_hand_pose'][:, -1, :],
                                    betas=res['mano_shape'][:, -1, :],
                                    transl=res['mano_trans'][:, -1, :])
            pred_joints = output.joints
            pred_vertices = output.vertices

            if config['log']['save_result']:
                batch_num = pred_joints.shape[0]
                K_tmp = annot['K'][0][0]
                joints_path = os.path.join(root_dir, seq_id, 'joints')
                mesh_path = os.path.join(root_dir, seq_id, 'mesh')
                mesh_pro_path = os.path.join(root_dir, seq_id, 'mesh_pro')
                img_path = os.path.join(root_dir, seq_id, 'imgs_2d_joints')
                eci_path = os.path.join(root_dir, seq_id, 'ecis')
                flow_path = os.path.join(root_dir, seq_id, 'flow')
                iwe_path = os.path.join(root_dir, seq_id, 'iwe')
                os.makedirs(joints_path, exist_ok=True)
                os.makedirs(mesh_path, exist_ok=True)
                os.makedirs(mesh_pro_path, exist_ok=True)
                os.makedirs(img_path, exist_ok=True)
                os.makedirs(eci_path, exist_ok=True)
                os.makedirs(flow_path, exist_ok=True)
                os.makedirs(iwe_path, exist_ok=True)
                for i in range(batch_num):
                    ## save pred 3d joints
                    joints = pred_joints[i].detach().cpu().numpy()[indices_change(1, 2)]
                    frame = x['ids'][i][-1].item()
                    joints_2d = joints @ (K_tmp.detach().cpu().numpy()).T
                    joints_2d = joints_2d[:,:2]/joints_2d[:,2:]
                    img = np.ones((config['data']['height'],config['data']['width'],3)) * 255.
                    for (u, v) in joints_2d:
                        img = cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255), -1)
                    if datasets[j].motion_type == 'fast':
                        cv2.imwrite(os.path.join(img_path, f'{fast_id}.jpg'), img)
                        json_write(os.path.join(joints_path, f'{fast_id}.json'), joints.tolist())
                    else:
                        cv2.imwrite(os.path.join(img_path, f'{frame}.jpg'), img)
                        json_write(os.path.join(joints_path, f'{frame}.json'), joints.tolist())

                    ## save pred 3d mesh
                    vertices = pred_vertices[i]
                    verts_rgb = torch.ones_like(vertices).unsqueeze(0).type_as(vertices)
                    textures = TexturesVertex(verts_rgb)
                    mesh = Meshes(
                        verts=vertices.unsqueeze(0),
                        faces=faces.unsqueeze(0),
                        textures=textures
                    )
                    R = torch.eye(3).to(vertices.device)
                    t = torch.zeros(3).to(vertices.device)
                    cameras = cameras_from_opencv_projection(
                        R=R.reshape(-1, 3, 3).type_as(vertices),
                        tvec=t.reshape(-1, 3).type_as(vertices),
                        camera_matrix=K_tmp.reshape(-1, 3, 3).type_as(vertices),
                        image_size=torch.tensor([config['data']['height'], config['data']['width']]).expand(1, 2).type_as(vertices)
                    ).to(vertices.device)
                    render.shader.to(vertices.device)
                    render_result = render(
                        mesh,
                        cameras=cameras
                    )
                    img = render_result[0,...,:3].detach().cpu().numpy()  * 255.

                    if datasets[j].motion_type == 'fast':
                        save_obj(os.path.join(mesh_path, f'{fast_id}.obj'), vertices, faces)
                        cv2.imwrite(os.path.join(mesh_pro_path, f'{fast_id}.jpg'),img)
                    else:
                        save_obj(os.path.join(mesh_path, f'{frame}.obj'), vertices, faces)
                        cv2.imwrite(os.path.join(mesh_pro_path, f'{frame}.jpg'),img)


                    ## save ECIs
                    eci = x['origin_eci'][i][-1].permute(1,2,0)
                    b_channel = torch.zeros((*eci.shape[:2],1),device=device).type_as(eci)
                    img_eci = torch.cat([eci,b_channel],dim=-1)
                    img_eci = img_eci.detach().cpu().numpy()[:,:,::-1] * 255.
                    if datasets[j].motion_type == 'fast':
                        if hand_type == 'right':
                            cv2.imwrite(os.path.join(eci_path, f'{fast_id}.jpg'), img_eci)
                        else:
                            cv2.imwrite(os.path.join(eci_path, f'{fast_id}.jpg'), cv2.flip(img_eci, 1))
                    else:
                        if hand_type == 'right':
                            cv2.imwrite(os.path.join(eci_path, f'{frame}.jpg'), img_eci)
                        else:
                            cv2.imwrite(os.path.join(eci_path, f'{frame}.jpg'), cv2.flip(img_eci, 1))

                    ## save pred flow
                    if config['method']['name'] != 'EventHands':
                        flow_i = (flow_pred[i, -1, :, :, -2:] / 8. * config['preprocess']['bbox']['size']).detach().cpu().numpy()
                        img_flow = flow_to_image(flow_i[:, :, 0], flow_i[:, :, 1]) * 255.
                        if datasets[j].motion_type == 'fast':
                            if hand_type == 'right':
                                cv2.imwrite(os.path.join(flow_path, f'{fast_id}.jpg'), img_flow[:, :, ::-1])
                            else:
                                cv2.imwrite(os.path.join(flow_path, f'{fast_id}.jpg'), cv2.flip(img_flow[:, :, ::-1], 1))
                        else:
                            if hand_type == 'right':
                                cv2.imwrite(os.path.join(flow_path, f'{frame}.jpg'), img_flow[:,:,::-1])
                            else:
                                cv2.imwrite(os.path.join(flow_path, f'{frame}.jpg'), cv2.flip(img_flow[:, :, ::-1], 1))

                    ## save IWE
                    if config['method']['name'] != 'EventHands':
                        if datasets[j].motion_type == 'fast':
                            iwe = get_iwe(config, x, res, i, device, K_tmp, mano_layer, render)
                            iwe = torch.cat([iwe, b_channel], dim=-1)
                            iwe = iwe.detach().cpu().numpy()[:, :, ::-1] * 255.
                            if hand_type == 'right':
                                cv2.imwrite(os.path.join(iwe_path, f'{fast_id}.jpg'), iwe)
                            else:
                                cv2.imwrite(os.path.join(iwe_path, f'{fast_id}.jpg'), cv2.flip(iwe, 1))

                        else:
                            frame_last = x['ids'][i][-2].item()
                            if frame == -1 or frame_last == -1:
                                continue
                            iwe = get_iwe(config, x, annot, i, device, K_tmp, mano_layer, render)
                            iwe = torch.cat([iwe, b_channel], dim=-1)
                            iwe = iwe.detach().cpu().numpy()[:, :, ::-1] * 255.
                            if hand_type == 'right':
                                cv2.imwrite(os.path.join(iwe_path, f'{frame}.jpg'), iwe)
                            else:
                                cv2.imwrite(os.path.join(iwe_path, f'{frame}.jpg'), cv2.flip(iwe, 1))

                    if datasets[j].motion_type == 'fast':
                        fast_id += 1

            gt_joints = annot['joint'][:, -1, :, :]
            #remove frames without ground truth labels
            ids = x['ids'][:,-1]
            valid_index = torch.where(ids!=-1)[0]
            if len(valid_index) == 0:
                continue
            pred_joints = pred_joints[valid_index]
            gt_joints = gt_joints[valid_index]

            if datasets[j].motion_type != 'fast':
                aligned_pred_joints = pred_joints - pred_joints[:,0:1,:]
                aligned_gt_joints = gt_joints - gt_joints[:,0:1,:]
                # align mpjpe
                mpjpe_error = torch.sqrt(torch.sum(torch.pow((aligned_pred_joints - aligned_gt_joints), 2), dim=-1))
                mpjpe_list.append(mpjpe_error.detach().cpu().numpy())
                mpjpe_batch = torch.mean(mpjpe_error,dim=-1)
                mpjpe_seq_error_list.extend(mpjpe_batch.detach().cpu().numpy().tolist())
                # PA-mpjpe
                pa_aligned_pred_joints = compute_similarity_transform_batch(aligned_pred_joints, aligned_gt_joints, device=device)
                pampjpe_error = torch.sqrt(torch.sum(torch.pow((pa_aligned_pred_joints - aligned_gt_joints), 2), dim=-1))
                pampjpe_batch = torch.mean(pampjpe_error,dim=-1)
                pampjpe_seq_error_list.extend(pampjpe_batch.detach().cpu().numpy().tolist())

                pred_joints_2d = torch.einsum('ijk,kl->ijl', pred_joints, K.T)
                pred_joints_2d = pred_joints_2d[..., :2] / pred_joints_2d[..., 2:]
                gt_joints_2d = torch.einsum('ijk,kl->ijl', gt_joints, K.T)
                gt_joints_2d = gt_joints_2d[..., :2] / gt_joints_2d[..., 2:]
                mpjpe_2d_each_joint = compute_2d_error(pred_joints_2d, gt_joints_2d, is_abs=True, is_align=True)
                mpjpe_2d_each_hand = torch.mean(mpjpe_2d_each_joint, dim=-1)
                #去掉异常值
                valid_mpjpe = torch.where(mpjpe_2d_each_hand<100)[0]
                mpjpe_2d_each_joint = mpjpe_2d_each_joint[valid_mpjpe]
                mpjpe_2d_each_joint_scale = compute_2d_error(pred_joints_2d, gt_joints_2d, is_abs=False, is_align=True)
                mpjpe_2d_each_joint_scale = mpjpe_2d_each_joint_scale[valid_mpjpe]
                mpjpe_2d = torch.mean(mpjpe_2d_each_joint)
                if torch.isnan(mpjpe_2d):
                    continue
                mpjpe_2d_scale = torch.mean(mpjpe_2d_each_joint_scale)
                mpjpe2d_seq_error_list.append(mpjpe_2d.item())
                mpjpe2d_seq_error_list_scale.append(mpjpe_2d_scale.item())

                # for batch_id, frame_id in enumerate(x['ids'][valid_index,-1].detach().cpu().numpy().tolist()):
                #     img = visualize_2d_error(pred_joints_2d[batch_id].detach().cpu().numpy(), gt_joints_2d[batch_id].detach().cpu().numpy())
                #     os.makedirs(os.path.join(root_dir, 'fast'), exist_ok=True)
                #     cv2.imwrite(os.path.join(root_dir, 'fast', 'img{}.jpg'.format(frame_id)), img)

            else:
                #2d-mpjpe
                frame_ids = x['ids'][valid_index,-1].detach().cpu().numpy().tolist()
                for batch_id, frame_id in enumerate(frame_ids):
                    if frame_id == -1 or str(frame_id) not in fast_annot['2d_joints']['event'].keys() or \
                            str(frame_id) not in fast_annot['3d_joints'].keys():
                        continue
                    joints_2d_gt = torch.tensor(fast_annot['2d_joints']['event'][str(frame_id)], dtype=torch.float32,device=device)
                    joints_3d_gt = torch.tensor(fast_annot['3d_joints'][str(frame_id)], dtype=torch.float32,device=device) / 1000.
                    joints_3d_gt = joints_3d_gt @ R.T + t
                    if hand_type == 'left':
                        root_distance_2d = 2*joints_2d_gt[0,0] - config['data']['width']
                        root_distance_3d = root_distance_2d * joints_3d_gt[0,2] / fx
                        root_3d_gt = joints_3d_gt[0].clone()
                        root_3d_gt[0] = root_3d_gt[0] - root_distance_3d
                        joints_2d_gt[:, 0] = config['data']['width'] - joints_2d_gt[:, 0]
                    else:
                        root_3d_gt = joints_3d_gt[0].clone()
                    joints_3d_pred = pred_joints[batch_id][indices_change(1,2)]
                    joints_3d_pred_align = joints_3d_pred# - joints_3d_pred[0:1] + root_3d_gt
                    joints_2d_pred = joints_3d_pred_align @ K.T
                    joints_2d_pred = joints_2d_pred[:,:2] / joints_2d_pred[:,2:]
                    mpjpe_2d_each_joint = compute_2d_error(joints_2d_pred.unsqueeze(0), joints_2d_gt.unsqueeze(0), is_abs=True, is_align=True)
                    mpjpe_2d_each_joint_scale = compute_2d_error(joints_2d_pred.unsqueeze(0), joints_2d_gt.unsqueeze(0),is_abs=False, is_align=True)
                    mpjpe_2d = torch.mean(mpjpe_2d_each_joint)
                    mpjpe_2d_scale = torch.mean(mpjpe_2d_each_joint_scale)
                    mpjpe2d_seq_error_list.append(mpjpe_2d.item())
                    mpjpe2d_seq_error_list_scale.append(mpjpe_2d_scale.item())
                    if config['log']['save_fast']:
                        img = visualize_2d_error(joints_2d_pred.detach().cpu().numpy(), joints_2d_gt.detach().cpu().numpy())
                        os.makedirs(os.path.join(root_dir, 'fast'), exist_ok=True)
                        cv2.imwrite(os.path.join(root_dir, 'fast', 'img{}.jpg'.format(frame_id)), img)
        if datasets[j].motion_type != 'fast':
            ## for auc calculation
            mpjpe_array = np.concatenate(mpjpe_list, axis=0)
            save_path = os.path.join(root_dir, seq_id)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, 'mpjpe.npy'), mpjpe_array)
            ##
            error_dict[scene+'_'+gesture]['mpjpe'].extend(mpjpe_seq_error_list)
            error_dict[scene + '_' + gesture]['pa-mpjpe'].extend(pampjpe_seq_error_list)
            error_dict[scene + '_' + gesture]['2d-mpjpe'].extend(mpjpe2d_seq_error_list)
            error_dict[scene + '_' + gesture]['2d-mpjpe_scale'].extend(mpjpe2d_seq_error_list_scale)
            error_dict['all_mpjpe']['mpjpe'].extend(mpjpe_seq_error_list)
            error_dict['all_mpjpe']['pa-mpjpe'].extend(pampjpe_seq_error_list)
            error_dict['all_mpjpe']['2d-mpjpe'].extend(mpjpe2d_seq_error_list)
            error_dict['all_mpjpe']['2d-mpjpe_scale'].extend(mpjpe2d_seq_error_list_scale)
            print('Sequence MPJPE: {}mm / PA-MPJPE: {}mm / 2D-MPJPE: {}px / 2D-MPJPE_scaled: {}px'.format(
                np.mean(mpjpe_seq_error_list)*1000., np.mean(pampjpe_seq_error_list)*1000.,
                np.mean(mpjpe2d_seq_error_list), np.mean(mpjpe2d_seq_error_list_scale)
            ))
        else:
            error_dict['fast']['2d-mpjpe'].extend(mpjpe2d_seq_error_list)
            error_dict['fast']['2d-mpjpe_scale'].extend(mpjpe2d_seq_error_list_scale)
            print('Sequence 2D-MPJPE: {}px / 2D-MPJPE_scaled: {}px'.format(np.mean(mpjpe2d_seq_error_list), np.mean(mpjpe2d_seq_error_list_scale)))
    print(50*'*')
    for k,v in error_dict.items():
        if k != 'fast':
            print('{} MPJPE: {}mm / PA-MPJPE: {}mm / 2D-MPJPE: {}px / 2D-MPJPE_scaled: {}px'.format(
                k, np.mean(v['mpjpe'])*1000., np.mean(v['pa-mpjpe'])*1000.,
                np.mean(v['2d-mpjpe']), np.mean(v['2d-mpjpe_scale'])
            ))
        else:
            print('{} 2D-MPJPE: {}px / 2D-MPJPE_scaled: {}px'.format(k, np.mean(v['2d-mpjpe']), np.mean(v['2d-mpjpe_scale'])))

    all_highlight_error = error_dict['highlight_fixed']['mpjpe'] + error_dict['highlight_random']['mpjpe']
    print(f"all highlight error: {np.mean(all_highlight_error) * 1000.}mm")

