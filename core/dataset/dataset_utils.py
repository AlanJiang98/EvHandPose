import os
import copy
from tools.basic_io.json_utils import json_read
import random


def get_dataset_info(config, mode='train'):
    '''
    :param config: config dict
    :param debug: bool whether used for debug
    :return: list of data direction of each sequence
    '''
    data_params = []
    if config['data']['dataset'] == 'EvHand':
        data_caps = [str(i) for i in range(84)]
        id_seq_all = json_read(os.path.join(config['data']['data_dir'], 'id_seq_all.json'))
        id_seq_right = id_seq_all["right"]
        id_seq_left = id_seq_all["left"]
        scenes = ["normal_fixed", "normal_random", "highlight_fixed", "highlight_random",
                  "flash_fixed", "flash_random", "fast"]
        if config['data']['flip']:
            id_seq = {}
            for key in id_seq_left.keys():
                id_seq[key] = [*id_seq_left[key], *id_seq_right[key]]
        else:
            if config['data']['hand_type'] == 'right':
                id_seq = id_seq_right
            else:
                id_seq = id_seq_left
        id_caps = []
        if mode != 'val':
            if mode == 'train':
                subject_ids = config['data']['train_subject']
            elif mode == 'eval':
                subject_ids = config['data']['eval_subject']
            else:
                pass
            for subject_id in subject_ids:
                id_caps = [*id_caps, *id_seq[str(subject_id)]]
            for cap in data_caps:
                if cap in id_caps:
                    tmp_dir = os.path.join(config['data']['data_dir'], cap)
                    data_params.append(tmp_dir)
        else:
            subject_ids = config['data']['train_subject']
            for scene in scenes:
                id_caps_temp = []
                for subject_id in subject_ids:
                    if str(subject_id) not in id_seq_all[scene].keys():
                        continue
                    id_caps_temp = [*id_caps_temp, *id_seq_all[scene][str(subject_id)]]
                cap = random.choice(id_caps_temp)
                tmp_dir = os.path.join(config['data']['data_dir'] , cap)
                data_params.append(tmp_dir)
                if scene == "normal_random":
                    cap = random.choice(id_caps_temp)
                    tmp_dir = os.path.join(config['data']['data_dir'], cap)
                    data_params.append(tmp_dir)

    else:
        pass

    return data_params


def get_dataset_configs(config, mode='train'):
    '''
    get configs for each sequence
    :param config: global config
    :return: config list
    '''
    data_params = get_dataset_info(config, mode)
    configs = []
    for param in data_params:
        config_tmp = copy.deepcopy(config)
        config_tmp['data']['seq_dir'] = param
        config_tmp['data']['tmp_cap'] = param.split('/')[-1]
        config_tmp['exper']['mode'] = mode
        configs.append(config_tmp)
    return configs


def datasets_preprocess(datasets):
    for i in range(len(datasets)-1, -1, -1):
        if datasets[i].len == 0:
            datasets.pop(i)
    return datasets