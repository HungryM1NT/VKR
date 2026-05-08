import collections
import collections.abc
collections.Sequence = collections.abc.Sequence

import os
import yaml
import mmrotate
from mmengine.config import Config
from mmengine.runner import Runner
from mmrotate.utils import register_all_modules

def get_classes_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return tuple(data['names'][k].replace(' ', '_') for k in sorted(data['names'].keys()))

def main():
    class_names = get_classes_from_yaml('training/YOLO_data/data.yaml')
    print(f"Классы для обучения: {class_names}")
    
    mmrotate_dir = os.path.dirname(mmrotate.__file__)
    config_dir = os.path.join(mmrotate_dir, '.mim', 'configs', 'rotated_faster_rcnn')
    
    possible_names = [
        'rotated_faster_rcnn_r50_fpn_1x_dota_le90.py',
        'rotated-faster-rcnn-r50-fpn_1x_dota-le90.py',
        'rotated-faster-rcnn-le90_r50_fpn_1x_dota.py'
    ]
    
    base_config_path = None
    for name in possible_names:
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            base_config_path = path
            break
            
    if base_config_path is None:
        raise FileNotFoundError("Конфиг не найден во внутренней папке mmrotate!")
        
    cfg = Config.fromfile(base_config_path)

    cfg.data_root = 'training/birdnet_dataset/'
    cfg.classes = class_names

    # Настройка Train
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.ann_file = 'train/labelTxt/'
    cfg.train_dataloader.dataset.data_prefix = dict(img_path='train/images/')
    cfg.train_dataloader.dataset.metainfo = dict(classes=cfg.classes)

    # Настройка Validation
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.ann_file = 'validation/labelTxt/'
    cfg.val_dataloader.dataset.data_prefix = dict(img_path='validation/images/')
    cfg.val_dataloader.dataset.metainfo = dict(classes=cfg.classes)

    # Настройка Test
    cfg.test_dataloader.dataset.data_root = cfg.data_root
    cfg.test_dataloader.dataset.ann_file = 'test/labelTxt/'
    cfg.test_dataloader.dataset.data_prefix = dict(img_path='test/images/')
    cfg.test_dataloader.dataset.metainfo = dict(classes=cfg.classes)

    cfg.model.roi_head.bbox_head.num_classes = len(cfg.classes)
    
    cfg.train_cfg.max_epochs = 36 
    cfg.train_dataloader.batch_size = 8
    cfg.train_dataloader.num_workers = 4
    cfg.optim_wrapper.optimizer.lr = 0.02
    
    cfg.work_dir = 'training/birdnet_work_dir'

    register_all_modules(init_default_scope=True)
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()