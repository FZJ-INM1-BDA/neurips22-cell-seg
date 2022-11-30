import celldetection as cd
from os.path import join, dirname, basename
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import neurips as cs
import os
from psutil import cpu_count
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def set_lr_(config, comm, rank, ranks):
    lr = config.lr
    if lr is None:
        base_lr = np.random.normal(**config.base_lr) if isinstance(config, dict) else config.base_lr
        if config.base_lr_sync:
            base_lr = comm.bcast(base_lr if rank == 0 else None, root=0)
        k = (config.batch_size * 4 * ranks) / (config.base_bs * 4)
        if config.base_lr_scale == 'linear':
            lr = base_lr * k
        elif config.base_lr_scale == 'sqrt':
            lr = base_lr * np.sqrt(k)
        else:
            raise ValueError(f'Not supported: {config.base_lr_scale}')
    if 'lr' not in config.optimizer:
        config.optimizer[list(config.optimizer.keys())[0]]['lr'] = lr


def set_sampler_seed_(config, comm, rank):
    if config.get('sampler_seed') is None:
        config.sampler_seed = np.random.randint(0, 999999999)
        if config.sampler_seed_sync:
            config.sampler_seed = comm.bcast(config.sampler_seed if rank == 0 else None, root=0)


def init_dist(config, rank, ranks, local_rank, local_ranks, device_count, device='cuda'):
    device_ids = [0]
    if config.distributed:
        master_addr = os.environ.get('MASTER_ADDR')
        master_port = os.environ.get('MASTER_PORT', 12343)
        if master_addr is None:
            raise ValueError('Specify MASTER_ADDR as environment variable to use distributed training.\n'
                             'For more information see: '
                             'https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#basic-use-case')

        n = device_count // local_ranks
        device_ids = list(range(local_rank * n, (local_rank + 1) * n))
        device = f'cuda:{device_ids[0]}'
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=ranks,
            rank=rank
        )
    return device_ids, device


def get_args(local_ranks):
    parser = argparse.ArgumentParser('Contour Proposal Networks for Instance Segmentation', add_help=False)
    parser.add_argument('-i', '--input_path', default='./inputs', type=str, help='Input path.')
    parser.add_argument('-o', '--output_path', default='./outputs', type=str, help='Output path.')
    parser.add_argument('-s', '--schedule', default='schedule.json', type=str, help='Schedule filename.')
    parser.add_argument('-c', '--config_id', default=0, type=int, help='Config index.')
    parser.add_argument('-w', '--workers', default=min(4, max(0, cpu_count(logical=False) // local_ranks - 4)),
                        type=int, help='Number of workers.')
    args = parser.parse_args()
    return args


def get_model(config, device, device_ids):
    model = cs.nn.build_cpn_model(config)
    model = model.to(device)
    if config.sync_batch_norm and config.distributed:
        print('Convert BatchNorm to SyncBatchNorm', flush=True)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)
    if config.distributed:
        print('Wrap model in DDP', flush=True)
        model = DDP(model, device_ids)
    return model


def set_start_epoch_(config):
    if config.checkpoint is None:
        config.start_epoch = 1
    else:
        bn = basename(config.checkpoint)
        assert 'epoch' in bn
        config.start_epoch = int(bn.split('_')[-1].replace('.pt', '')) + 1  # assuming naming scheme: *epoch_100.pt


def get_optimizer(config, model, num_data_points):
    optimizer = cd.conf2optimizer(config.optimizer, model.parameters())
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    if next(iter(config.scheduler.keys())) == 'WarmupMultiStepLR':
        milestones = [int(f * config.epochs * num_data_points) for f in config.scheduler_milestones_as_fractions]
        scheduler = cs.nn.WarmupMultiStepLR(optimizer, milestones)
    else:
        scheduler = cd.conf2scheduler(config.scheduler, optimizer)
    return optimizer, scheduler, scaler


def train(config, model, rank, sampler, optimizer, scheduler, scaler, writer, device, train_loader):
    # Train
    global_step = 0
    for epoch in range(config.start_epoch, config.epochs + 1):
        if rank == 0:
            print(f'(rank {rank})  Epoch: {epoch}/{config.epochs}', flush=True)
        if config.distributed and sampler is not None and isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        global_step = cs.train_epoch(
            model=model, train_loader=train_loader, global_step=global_step, device=device, optimizer=optimizer,
            desc=f'Epoch {epoch}/{config.epochs}', scaler=scaler, scheduler=scheduler,
            scheduler_on_step=config.scheduler_on_step, gpu_stats=config.show_gpu_stats, progress=config.show_progress,
            writer=writer
        )

        if epoch % config.save_frequency == 0 and epoch >= config.save_min_epoch:
            cs.save(model, join(config.directory, f'{config.model}_epoch_%0{len(str(config.epochs))}d.pt' % epoch),
                    config=config, rank=rank)

        if scheduler is not None and not config.scheduler_on_step:
            scheduler.step()


def get_writer(config):
    kw = config.writer_kwargs
    if kw is None:
        return None
    kw['comment'] = config.model_tag + '_' * (len(kw.get('comment', '')) > 0) + kw.get('comment', '')
    return SummaryWriter(**kw)


def neurips_handler(img, labels, meta):
    return img, labels, meta


def external_handler(meta, content):
    img, labels = content
    return img, labels, meta


def pseudo_labels_handler(img, labels, contours, scores, boxes, uncertainty, score_maps, meta):
    return img, labels, contours, scores, boxes, uncertainty, score_maps, meta


def main():
    try:
        comm, rank, ranks = cd.mpi.get_comm(None, True)
        local_comm, local_rank, local_ranks = cd.mpi.get_local_comm(comm, True)
    except:
        comm, rank, ranks = local_comm, local_rank, local_ranks = None, 0, 1

    args = get_args(local_ranks=local_ranks)

    schedule = cd.Schedule()
    schedule.load(args.schedule)
    config = schedule[args.config_id]

    config.model_tag, config.directory = cd.random_code_name_dir(args.output_path)
    device_count = torch.cuda.device_count()
    assert device_count >= 1, 'No cuda devices found :('
    writer = get_writer(config)
    config.distributed = ranks > 1

    device_ids, device = init_dist(config, rank, ranks, local_rank, local_ranks, device_count=device_count,
                                   device='cuda')
    set_lr_(config, comm, rank, ranks)
    set_sampler_seed_(config, comm, rank)

    print(f'(rank {rank}):', config, flush=True)

    # Data
    gray_transforms = cs.get_aug(config.aug_plan, rgb=False, crop_size=config.crop_size)
    rgb_transforms = cs.get_aug(config.aug_plan, rgb=True, crop_size=config.crop_size)

    print('Neurips data:', join(args.input_path, 'neurips_data'))
    neurips_data = cs.NeurIpsTrainLabeled(join(args.input_path, 'neurips_data'), rgb=False, cache=False,
                                          norm_method=config.data_norm_method)
    print('External data:', join(args.input_path, 'external_data'))
    external_data = cd.data.GenericH5(glob(join(args.input_path, 'external_data', '*.h5')), ('image', 'labels'))
    data_objects = [external_data]
    data_handlers = [external_handler]
    for _ in range(config.neurips_reps):
        data_objects.append(neurips_data)
        data_handlers.append(neurips_handler)

    if config.pseudo_labels is not None:
        pl = cs.PseudoLabels(config.pseudo_labels, items=config.pseudo_labels_num)
        data_objects.append(pl)
        data_handlers.append(pseudo_labels_handler)

    train_composition = cs.Composition(*data_objects, handlers=data_handlers)
    train_data = cs.data.Dataset(
        train_composition, config,
        gray_transforms, rgb_transforms,
        size=config.crop_size
    )

    kw = dict(shuffle=config.shuffle)
    if args.workers:
        kw['prefetch_factor'] = config.prefetch_factor
    sampler = None
    if config.distributed:
        sampler = DistributedSampler(train_data, num_replicas=ranks, rank=rank, shuffle=config.shuffle,
                                     seed=config.sampler_seed)
        kw['sampler'] = sampler
        kw['shuffle'] = False

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, num_workers=args.workers,
        collate_fn=cd.universal_dict_collate_fn,
        pin_memory=config.pin_memory, **kw
    )

    model = get_model(config, device, device_ids)
    optimizer, scheduler, scaler = get_optimizer(config, model, len(train_data))
    set_start_epoch_(config)
    train(config, model, rank, sampler, optimizer, scheduler, scaler, writer, device, train_loader)


if __name__ == "__main__":
    main()
