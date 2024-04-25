import os
import torch
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from core.model.smplx.body_models import MANO
import cv2

def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T
    return S1_hat

def compute_similarity_transform_batch(S1, S2, device):
    """Batched version of compute_similarity_transform."""

    if type(S1) == torch.Tensor:
        S1 = S1.detach().cpu().numpy()
        S2 = S2.detach().cpu().numpy()
        is_tensor = True
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    S1_hat = torch.tensor(S1_hat, device=device)
    return S1_hat

def compute_2d_kps(joints_3d, K):
    kps_2d = torch.bmm(K, joints_3d.permute(0, 2, 1)).permute(0, 2, 1)
    kps_2d[:, :, :2] = kps_2d[:, :, :2] / kps_2d[:, :, 2:]
    return kps_2d[:, :, :2]


def compute_2d_error(kps_2d_pred, kps_2d_gt, is_abs=False, is_align=True):
    if is_align:
        kps_p = kps_2d_pred - kps_2d_pred[:, :1] + kps_2d_gt[:, :1]
    else:
        kps_p = kps_2d_pred
    if is_abs:
        kps_error = torch.sqrt(torch.sum((kps_p - kps_2d_gt)**2, dim=-1))
    else:
        kps_p_scale = kps_p / torch.sqrt(torch.sum((kps_p[:, 9:10] - kps_p[:, 0:1])**2, keepdim=True, dim=-1) + 1e-3)
        kps_gt_scale = kps_2d_gt / torch.sqrt(torch.sum((kps_2d_gt[:, 9:10] - kps_2d_gt[:, 0:1])**2, keepdim=True, dim=-1) + 1e-3)
        kps_error = torch.sqrt(torch.sum((kps_p_scale - kps_gt_scale)**2, dim=-1))
    return kps_error

def visualize_2d_error(kps_2d_pred, kps_2d_gt):
    img = np.ones((260, 346, 3)) * 255.
    for (x, y) in kps_2d_gt:
        img = cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
    for (x, y) in kps_2d_pred:
        img = cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    return img


def get_kalman_filter(config, size):
    filter = KalmanFilter(dim_x=2 * size, dim_z=size)
    filter.F = np.zeros((2 * size, 2 * size))
    filter.H = np.zeros((size, 2 * size))
    filter.R *= config['method']['kalman_filter']['R_coef']
    filter.Q = Q_discrete_white_noise(2, 1. / config['preprocess']['test_fps'],
                                      config['method']['kalman_filter']['var'],
                                      block_size=size)
    for i in range(size):
        filter.F[i, i] = 1.
        filter.F[i, i+size] = 1. / config['preprocess']['test_fps']
        filter.F[i+size, i+size] = 1.
        filter.H[i, i] = 1
    return filter


def update_kalman_filter(filter, mano_now, init=False):
    mano_params = ['mano_hand_pose', 'mano_trans', 'mano_shape', 'mano_rot_pose']
    mano_dim = [0, 45, 48, 58, 61]
    data = torch.cat([mano_now[key][:, -1] for key in mano_params], dim=1).squeeze()
    data_np = data.cpu().numpy()
    if init:
        filter.x = np.r_[data_np, np.ones_like(data_np) * 0.05]
        mano_new = {}
        for key in mano_params:
            mano_new[key] = mano_now[key][:, -1].clone()
        return filter, mano_new
    else:
        filter.predict()
        filter.update(data_np)
        X = filter.x
        mano_np = X[:61]
        mano_new = {}
        for i, key in enumerate(mano_params):
            mano_new[key] = torch.tensor(mano_np[mano_dim[i]:mano_dim[i+1]][None, ...], dtype=torch.float32)
        return filter, mano_new


def get_mano_joints(model, mano_np, device):
    mano_params = ['mano_hand_pose', 'mano_trans', 'mano_shape', 'mano_rot_pose']
    mano_dim = [0, 45, 48, 58, 61]
    manos = {}
    for i, key in enumerate(mano_params):
        manos[key] = torch.tensor(mano_np[mano_dim[i]:mano_dim[i + 1]][None, ...], dtype=torch.float32).to(device)
    output = model(
        global_orient=manos['mano_rot_pose'].reshape(-1, 3),
        hand_pose=manos['mano_hand_pose'].reshape(-1, 45),
        betas=manos['mano_shape'].reshape(-1, 10),
        transl=manos['mano_trans'].reshape(-1, 3)
    )
    joints = output.joints.reshape(21, 3).detach().cpu().numpy()
    return joints


def filter_mano(config, manos, joints, device):
    size = 61
    filter = KalmanFilter(dim_x=2 * size, dim_z=size)
    filter.F = np.zeros((2 * size, 2 * size))
    filter.H = np.zeros((size, 2 * size))
    filter.R *= config['method']['kalman_filter']['R_coef']
    filter.Q = Q_discrete_white_noise(2, 1. / config['preprocess']['test_fps'],
                                      config['method']['kalman_filter']['var'],
                                      block_size=size)
    for i in range(size):
        filter.F[i, i] = 1.
        filter.F[i, i + size] = 1. / config['preprocess']['test_fps']
        filter.F[i + size, i + size] = 1.
        filter.H[i, i] = 1

    smplx_path = config['data']['smplx_path']
    mano_layer = {'right': MANO(smplx_path, ir_rhand=True, use_pca=False),
                  'left': MANO(smplx_path, ir_rhand=False, use_pca=False)}
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        mano_layer['left'].shapedirs[:, 0, :] *= -1
    if config['data']['flip']:
        mano_model = mano_layer['right'].to(device)
    else:
        mano_model = mano_layer[config['data']['hand_type']].to(device)
    uncertainty = np.zeros(config['method']['kalman_filter']['seq_len'])

    mano_params = ['mano_hand_pose', 'mano_trans', 'mano_shape', 'mano_rot_pose']
    mano_dim = [0, 45, 48, 58, 61]
    if len(manos['mano_hand_pose'].shape) == 3:
        data = torch.cat([manos[key][:, -1] for key in mano_params], dim=1).squeeze()
    else:
        data = torch.cat([manos[key] for key in mano_params], dim=1).squeeze()
    data_np = data.cpu().numpy()
    mano_filtered_np = np.zeros_like(data_np)
    mano_filtered_np[0] = data_np[0]
    filter.x = np.r_[data_np[0], np.ones_like(data_np[0]) * 0.01]
    joints_ = joints
    if len(manos['mano_hand_pose'].shape) == 4:
        joints_ = joints[:, -1]
    joints_align = (joints_ - joints_[:, :1]).detach().cpu().numpy()
    uncertainty_all = np.zeros(data_np.shape[0])


    count = 0
    for i in range(1, len(data_np)):
        if i < config['method']['kalman_filter']['seq_len']:
            filter.predict()
            x_tmp = filter.x[:61]

            joints_tmp = get_mano_joints(mano_model, x_tmp, device)
            uncertainty[i] = np.sum(np.sqrt(np.sum((joints_tmp - joints_tmp[:1] - joints_align[i])**2, axis=-1))) / 20.
            uncertainty_all[i] = uncertainty[i]

            filter.update(data_np[i])
            x = filter.x
            mano_filtered_np[i] = x[:61]
        else:
            filter.predict()
            x_tmp = filter.x[:61]
            joints_tmp = get_mano_joints(mano_model, x_tmp, device)
            uncer_tmp = np.sum(np.sqrt(np.sum((joints_tmp - joints_tmp[:1] - joints_align[i]) ** 2, axis=-1))) / 20.
            uncertainty_all[i] = uncer_tmp
            uncer_mean = np.mean(uncertainty)
            uncer_std = max(np.std(uncertainty), config['method']['kalman_filter']['min_std'])

            if uncer_tmp > 0.45 * (1 + 0.25)**count: # uncer_mean + uncer_std * config['method']['kalman_filter']['sigma']:
                print('%%%%%%%%%%%%%%%%%%%%')
                print('item: {} uncer: {} thres: {} std: {}'.format(i, uncer_tmp, uncer_mean + uncer_std * config['method']['kalman_filter']['sigma'], np.std(uncertainty)))
                mano_filtered_np[i] = x_tmp
                filter.update(x_tmp)
                uncertainty = np.r_[uncertainty[1:], uncer_tmp]
                count += 1
            else:
                filter.update(data_np[i])
                mano_filtered_np[i] = filter.x[:61]
                joints_tmp = get_mano_joints(mano_model, mano_filtered_np[i], device)
                uncer_tmp = np.sum(np.sqrt(np.sum((joints_tmp - joints_tmp[:1] - joints_align[i]) ** 2, axis=-1))) / 20.
                uncertainty = np.r_[uncertainty[1:], uncer_tmp]
                count = 0
    mano_filtered = {}
    for i, key in enumerate(mano_params):
        mano_filtered[key] = torch.tensor(mano_filtered_np[:, mano_dim[i]:mano_dim[i + 1]], dtype=torch.float32)
    return mano_filtered, uncertainty_all


def print_mpjpes(error_list, labels_list, f=None):
    errors_all = 0
    joints_all = 0
    # remove the fast motion results
    for i in range(len(error_list)):
        if error_list[i] is None:
            if f is None:
                print('{} is {}'.format(labels_list[i], None))
            else:
                f.write('{} is {}'.format(labels_list[i], None) + '\n')
        else:
            valid_joint = error_list[i] != 0.
            mpjpe_tmp = torch.sum(error_list[i]) / (torch.sum(valid_joint)+1e-6)
            if i != len(error_list) - 1:
                errors_all += torch.sum(error_list[i])
                joints_all += torch.sum(valid_joint)
            if f is None:
                print('{} is {}'.format(labels_list[i], mpjpe_tmp))
            else:
                f.write('{} is {}'.format(labels_list[i], mpjpe_tmp) + '\n')
    mpjpe_all = errors_all / (joints_all+1e-6)
    if f is None:
        print('MPJPE all is {}'.format(mpjpe_all))
    else:
        f.write('MPJPE all is {}'.format(mpjpe_all) + '\n')


def plot_3D_PCK(all_joints_error_: list, colors:list, labels:list, dir:str, filename: str, name='the'):
    '''
    plot 3D PCK and return AUC list
    :param all_joints_error: list of all joints error
    :param colors: color list
    :param labels: label list
    :param dir: output direction
    :return: list of AUCs
    '''
    assert len(all_joints_error_) == len(colors)
    font_size = 18
    x_max_percent = 0.98
    x_start = 0.
    x_end = 100000.
    all_joints_error = []
    # print(len(all_joints_error_))

    for i in range(len(all_joints_error_)):
        # print(all_joints_error_[i].shape)
        all_joints_error.append(all_joints_error_[i][:, 1:].clone())
        all_joints_error[i], indices = torch.sort(all_joints_error[i].reshape(-1))
        x_end_tmp = all_joints_error[i][int(x_max_percent * len(all_joints_error[i]))]
        if x_end_tmp < x_end:
            x_end = x_end_tmp
    step = 0.1
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    x_end = 100.
    ax.set_xlim((x_start, x_end))
    ax.set_ylim((0., 1.))
    ax.grid(True)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(font_size)

    legend_labels = []
    lines = []
    AUCs = []

    for method_id in range(len(all_joints_error)):
        color = colors[method_id]
        errors = all_joints_error[method_id]
        x_axis = torch.arange(x_start, x_end, step)
        pcks = torch.searchsorted(errors, x_axis) / errors.shape[0]
        AUC = torch.sum((pcks * step) / (x_end - x_start))
        AUCs.append(AUC)
        label = labels[method_id]
        label += ' AUC:%.03f' % AUC
        line, = ax.plot(x_axis, pcks, color=color, linewidth=2)
        lines.append(line)
        legend_labels.append(label)

    legend_location = 4
    ax.legend(lines, legend_labels, loc=legend_location, title_fontsize=font_size-2,
               prop={'size': font_size-2})
    ax.set_xlabel('error (mm)', fontsize=font_size)
    ax.set_title('3D PCK on ' + name + ' sequences', fontsize=font_size+4, pad=30)
    # plt.ylabel('3D-PCK', fontsize=font_size)
    plt.tight_layout()
    # plt.show()
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, filename))
    fig.clear()
    return AUCs