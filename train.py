import warnings
import argparse
import fnmatch
from pathlib import Path
import torch
from torch.nn.parameter import UninitializedParameter
from datasets.data_utils import get_dataloaders
from trainer.trainer import Trainer
from utils.config import ConfigNode
from utils.config_loader import apply_dotlist_overrides, load_composed_config
from utils.factory import instantiate
from utils.init_utils import set_random_seed, setup_saving_and_logging
warnings.filterwarnings('ignore', category=UserWarning)

def _match_param(name, patterns):
    return any(fnmatch.fnmatch(name, pattern) or pattern in name for pattern in patterns)

def _format_param_count(named_params):
    initialized = 0
    lazy = 0
    for _, param in named_params:
        if isinstance(param, UninitializedParameter):
            lazy += 1
            continue
        initialized += param.numel()
    if lazy:
        return f'{initialized:,} initialized params + {lazy} lazy params'
    return f'{initialized:,} params'

def build_optimizer(model, config, logger):
    named_params = [(name, param) for (name, param) in model.named_parameters() if param.requires_grad]
    if not named_params:
        raise RuntimeError('No trainable model parameters found.')
    group_cfgs = list(config.optimizer.get('param_groups', []))
    if not group_cfgs:
        logger.info(f'Trainable parameters: {_format_param_count(named_params)}')
        return instantiate(config.optimizer, params=[param for _, param in named_params])

    assigned = set()
    param_groups = []
    base_lr = float(config.optimizer.get('lr', 1.0e-3))
    base_weight_decay = float(config.optimizer.get('weight_decay', 0.0))
    for group_cfg in group_cfgs:
        patterns = list(group_cfg.get('patterns', []))
        if not patterns:
            continue
        group_named = [
            (name, param)
            for (name, param) in named_params
            if id(param) not in assigned and _match_param(name, patterns)
        ]
        if not group_named:
            logger.warning(f"Optimizer param group '{group_cfg.get('name', patterns)}' matched no trainable parameters.")
            continue
        for _, param in group_named:
            assigned.add(id(param))
        group = {'params': [param for _, param in group_named]}
        group['lr'] = float(group_cfg.get('lr', base_lr * float(group_cfg.get('lr_mult', 1.0))))
        group['weight_decay'] = float(group_cfg.get('weight_decay', base_weight_decay))
        param_groups.append(group)
        logger.info(
            f"Optimizer group '{group_cfg.get('name', ','.join(patterns))}': "
            f"{_format_param_count(group_named)}, "
            f"lr={group['lr']}, weight_decay={group['weight_decay']}"
        )

    remaining = [(name, param) for (name, param) in named_params if id(param) not in assigned]
    if remaining:
        param_groups.append({'params': [param for _, param in remaining], 'lr': base_lr, 'weight_decay': base_weight_decay})
        logger.info(
            f"Optimizer group 'default': {_format_param_count(remaining)}, "
            f"lr={base_lr}, weight_decay={base_weight_decay}"
        )
    return instantiate(config.optimizer, params=param_groups)

def main(config_path, overrides=None):
    raw_config = load_composed_config(config_path)
    raw_config = apply_dotlist_overrides(raw_config, overrides or [], config_dir=Path(config_path).parent)
    config = ConfigNode(raw_config)
    set_random_seed(config.trainer.seed)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger=logger, config=config)
    if config.trainer.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config.trainer.device
    (dataloaders, batch_transforms) = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    logger.info(model)
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)
    optimizer = build_optimizer(model, config, logger)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    epoch_len = config.trainer.get('epoch_len')
    trainer = Trainer(model=model, criterion=loss_function, metrics=metrics, optimizer=optimizer, lr_scheduler=lr_scheduler, config=config, device=device, dataloaders=dataloaders, epoch_len=epoch_len, logger=logger, writer=writer, batch_transforms=batch_transforms, skip_oom=config.trainer.get('skip_oom', True))
    trainer.train()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yaml')
    (args, unknown) = parser.parse_known_args()
    main(args.config, overrides=unknown)
