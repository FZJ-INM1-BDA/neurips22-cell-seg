import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from os.path import join, dirname, basename, isfile, isdir
from tqdm import tqdm
import celldetection as cd
from celldetection import GpuStats
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os

__all__ = ['barrier', 'train_epoch', 'save']


def print_to_file(*args, filename, mode='w', **kwargs):
    with open(filename, mode=mode) as f:
        print(*args, file=f, **kwargs)


def barrier():
    if dist.is_initialized():  # just an is None check
        dist.barrier()


def set_desc(tq, desc, loss, losses=None, scheduler=None, gpu_st=None):
    info = [] if desc is None else [desc]
    if gpu_st is not None:
        info.append(str(gpu_st))
    if losses is not None and isinstance(losses, dict):
        info.append('losses(' + ', '.join(
            [(f'{k}: %g' % np.round(cd.asnumpy(v), 3)) for k, v in losses.items() if v is not None]) + ')')
    if scheduler is not None:
        last_lr = getattr(scheduler, '_last_lr', None)
        if last_lr is not None:
            info += [f'lr: {last_lr}']
    info.append('loss %g' % np.round(cd.asnumpy(loss), 3))
    tq.desc = ' - '.join(info)


def add2writer(writer, global_step, loss, losses, scheduler=None):
    writer.add_scalar(f'loss/loss', loss, global_step=global_step)
    if losses is not None:
        for k, v in losses.items():
            if v is None:
                continue
            try:
                writer.add_scalar(f'loss/{k}', v, global_step=global_step)
            except Exception as e:
                print(e, k, v)
    if scheduler is not None:
        last_lr = None
        try:
            last_lr = getattr(scheduler, '_last_lr', None)
            if last_lr is not None:
                last_lr = float(np.array(last_lr))
                writer.add_scalar('lr', last_lr, global_step=global_step)
        except Exception as e:
            print(e, last_lr, flush=True)


def train_epoch(model, train_loader, global_step, device, optimizer, desc=None, scaler=None,
                scheduler=None, scheduler_on_step=True, gpu_stats=True, progress=True, writer=None):
    model.train()
    tq = tqdm(train_loader, desc=desc) if progress else train_loader
    gpu_st = None
    if gpu_stats:
        gpu_st = GpuStats()

    for batch_idx, batch in enumerate(tq):
        losses = {}
        batch: dict = cd.to_device(batch, device)
        if global_step % 1000 == 0 and writer is not None:
            writer.add_images('batch_inputs', batch['inputs'], global_step=global_step)
        optimizer.zero_grad()
        with autocast(scaler is not None):
            outputs: dict = model(batch['inputs'], targets=batch)
        loss = outputs['loss']
        losses.update(outputs.get('losses', {})), losses.update(outputs.get('info', {}))

        barrier()
        if writer is not None:
            add2writer(writer, global_step, loss=loss, losses=losses, scheduler=scheduler)
        if progress:
            set_desc(tq, desc, loss, losses, scheduler=scheduler, gpu_st=gpu_st)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None and scheduler_on_step:
            scheduler.step()

        global_step += 1
    return global_step


def save(model, filename, model_txt='model.txt', config=None, meta: dict = None, rank=None):
    print('Saving model to', filename, flush=True)
    if isinstance(model, DDP):
        model = model.module

    if rank is None or rank == 0:
        torch.save(model, filename)
        if model_txt:
            model_txt = join(dirname(filename), model_txt)
            if not os.path.isfile(model_txt):
                with open(model_txt, 'w') as f:
                    print(model, file=f)
    if config is not None:
        conf_filename = join(dirname(filename), 'config' + (f'_r{rank}' * (rank is not None)) + '.json')
        if not isfile(conf_filename) and isinstance(config, cd.Config):
            try:
                config.to_json(conf_filename)
                print_to_file(config, filename=join(dirname(filename),
                                                    'config' + (f'_r{rank}' * (rank is not None)) + '.txt'))
            except Exception as e:
                print(e)
    if meta is not None:
        meta_filename = join(dirname(filename), 'meta.json')
        if not isfile(meta_filename):
            try:
                cd.to_json(join(dirname(filename), 'meta.json'), meta)
            except Exception as e:
                print(e)
