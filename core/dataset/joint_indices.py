import numpy as np
# this scripts aims to deal with the skeleton correspondence

# from_interhand_joints_to_mano = [
#         20, 7, 6, 5, 11, 10, 9, 19, 18, 17, 15, 14, 13, 3, 2, 1, 0, 4, 8, 12, 16
#     ]
#
# from_mano_joints_to_interhand = [
#         16, 15, 14, 13, 17, 3, 2, 1, 18, 6, 5, 4, 19, 12, 11, 10, 20, 9, 8, 7, 0
# ]

colors = [
    'r', 'g', 'b', 'c', 'm', 'y', 'orange'
]

def indices_change(src=0, dst=1):
    '''
    This function are used to give the correspondences of joint indices from
    different dataset.
    0: interhand dataset, 1: smplx dataset, 2: our real dataset
    Usage:
    if you have interhand joints: J and you want to change their orders to smplx,
    J_smplx = J_interhand[indices_change(0, 1)]
    :param src: source joints order
    :param dst: destination joints order
    :return:
    '''
    skeleton_ids = {
        'root': [20, 0, 0],
        'thumb1': [3, 13, 1],
        'thumb2': [2, 14, 2],
        'thumb3': [1, 15, 3],
        'thumb4': [0, 16, 4],
        'index1': [7, 1, 5],
        'index2': [6, 2, 6],
        'index3': [5, 3, 7],
        'index4': [4, 17, 8],
        'middle1': [11, 4, 9],
        'middle2': [10, 5, 10],
        'middle3': [9, 6, 11],
        'middle4': [8, 18, 12],
        'ring1': [15, 10, 13],
        'ring2': [14, 11, 14],
        'ring3': [13, 12, 15],
        'ring4': [12, 19, 16],
        'pinky1': [19, 7, 17],
        'pinky2': [18, 8, 18],
        'pinky3': [17, 9, 19],
        'pinky4': [16, 20, 20]
    }
    indices = np.array(list(skeleton_ids.values()))
    raw = list(range(21))
    src_id = indices[:, src].tolist()
    dst_id = indices[:, dst].tolist()
    tmp = [dst_id.index(i) for i in raw]
    res = [src_id[i] for i in tmp]
    return res