import os
import sys
from os.path import dirname
abs_path = os.path.abspath(dirname(dirname(__file__)))
sys.path.insert(0,abs_path)
import argparse
import copy
from configs.parser import YAMLParser
import multiprocessing as mp
import warnings
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from core.dataset.dataset_utils import get_dataset_configs, datasets_preprocess
from core.dataset.dataset import EvHandDataset
from core.model.models import EvHands

def collect_train_datasets(dataset):
    global train_datasets
    train_datasets.append(dataset)

def collect_val_datasets(dataset):
    global val_datasets
    val_datasets.append(dataset)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser('Train Script')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--flow_model_path', type=str, default='')
    parser.add_argument('--config_merge', type=str, default='')
    args = parser.parse_args()
    config = YAMLParser(args.config)

    if args.config_merge != '':
        config.merge_configs(args.config_merge)
    config = config.config
    if args.model_path != '':
        config['method']['model_path'] = args.model_path
    if args.gpus != '':
        config['exper']['gpus'] = args.gpus
    if not config['data']['flip']:
        config['exper']['exper_name'] = config['exper']['exper_name'] + '_' + config['data']['hand_type']
    if config['method']['flow']['fixed']:
        if 'flow_encoder' in config['method']['optimizers'].keys():
            config['method']['optimizers'].pop('flow_encoder')
    else: #only train flow model
        if 'infer_encoder' in config['method']['optimizers'].keys():
            config['method']['optimizers'].pop('infer_encoder')

    ## get configs
    train_configs = get_dataset_configs(config, 'train')
    val_configs = get_dataset_configs(config, 'val')

    ## remove repeat configs in train_configs
    print(len(train_configs))
    val_seq_dirs = [config_tmp['data']['seq_dir'] for config_tmp in val_configs]
    train_configs = [config_tmp for config_tmp in train_configs if config_tmp['data']['seq_dir'] not in val_seq_dirs]
    print(len(train_configs))

    train_datasets = []
    val_datasets = []
    if config['exper']['debug']:
        for i, config_ in enumerate(train_configs):
            train_datasets.append(eval(config['data']['dataset']+'Dataset')(config_))
            print(config_['data']['seq_dir'])
            if len(train_datasets) > 0:
                break
        for i, config_ in enumerate(val_configs):
            val_datasets.append(eval(config['data']['dataset']+'Dataset')(config_))
            print(config_['data']['seq_dir'])
            if len(val_datasets) > 0:
                break
    else:
        pool = mp.Pool(8)
        for i, config_ in enumerate(train_configs):
            print(config_['data']['seq_dir'])
            pool.apply_async(eval(config['data']['dataset'] + 'Dataset'), args=(config_,), callback=collect_train_datasets)
        for i, config_ in enumerate(val_configs):
            pool.apply_async(eval(config['data']['dataset'] + 'Dataset'), args=(config_,), callback=collect_val_datasets)
        pool.close()
        pool.join()

    train_datasets = datasets_preprocess(train_datasets)
    val_datasets = datasets_preprocess(val_datasets)
    datasets = train_datasets + val_datasets

    for i in range(len(datasets)):
        datasets[i].set_index(i)

    print('train dataset: ', len(train_datasets))
    print('val dataset: ', len(val_datasets))
    train_data = ConcatDataset(train_datasets)
    val_data = ConcatDataset(val_datasets)

    train_loader = DataLoader(train_data, batch_size=config['preprocess']['batch_size'], shuffle=True, num_workers=config['preprocess']['num_workers'], #persistent_workers=True,
                              pin_memory=False)
    val_loader = DataLoader(val_data, batch_size=config['preprocess']['batch_size'], num_workers=config['preprocess']['num_workers'], #persistent_workers=True,
                             pin_memory=False)
    device = f'cuda:{args.gpus}'
    if config['method']['model_path'] is not None:
        model = eval(config['method']['name']).load_from_checkpoint(checkpoint_path=config['method']['model_path'],
                                                                    config=copy.deepcopy(config),map_location=device)
        print('model loaded! {}'.format(config['method']['model_path']))
    else:
        model = eval(config['method']['name'])(copy.deepcopy(config))
        print('model loaded!')
    if args.flow_model_path != '':
        config['method']['flow']['model_path'] = args.flow_model_path
        config_tmp = copy.deepcopy(config)
        config_tmp['method']['flow'].pop('last_for_output')
        flow_model = eval(config['method']['name']).load_from_checkpoint(checkpoint_path=config['method']['flow']['model_path'],
                                                                    config=config_tmp,map_location=device)
        model.flow_encoder = flow_model.flow_encoder
        print('set the flow model!')

    # keep dataset index
    if config['method']['name'] == 'EvHands':
        model.dataset = datasets

    output_path = os.path.join(config['exper']['output_dir'], config['exper']['exper_name'])
    os.makedirs(output_path, exist_ok=True)
    YAMLParser.save_config_dict(config, output_path, 'train.yml')

    loggers, callbacks = None, []
    loggers = TensorBoardLogger(save_dir=output_path, name='logs')
    metric = 'val_mpjpe' if not config['method']['flow']['train'] else 'val_flow_loss'
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename=config['method']['name'] + '_{epoch:02d}_{' + metric + ':.4f}',
        monitor=metric,
        save_top_k=100,
        save_last=True,
        verbose=True,
        mode='min')
    callbacks.append(checkpoint_callback)


    if config['exper']['early_stop']:
        earlystop_callback = EarlyStopping(
            monitor=metric,
            mode='min',
            patience=config['exper']['patience'], )
        callbacks.append(earlystop_callback)
    trainer = pl.Trainer(
        devices=[int(args.gpus)],
        max_epochs=config['exper']['epochs'],
        check_val_every_n_epoch=1,
        strategy=config['exper']['strategy'],
        logger=loggers,
        callbacks=callbacks,
        num_sanity_val_steps=0
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

