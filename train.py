import os
import argparse
import json
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tqdm import tqdm

from logger import Logger
from model import Inpainter
from data import MelSpecDataset, ChannelMasking
from utils import ConfigWrapper, show_message, str2bool


def run_training(rank, config, args):
    if args.n_gpus > 1:
        init_distributed(rank, args.n_gpus, config.dist_config)
        torch.cuda.set_device(f'cuda:{rank}')

    show_message('Initializing logger...', verbose=args.verbose, rank=rank)
    logger = Logger(config, rank=rank)

    show_message('Initializing model...', verbose=args.verbose, rank=rank)
    model = Inpainter(config).cuda()
    show_message(f'Number of CHANNEL INPAINTER parameters: {model.nparams}', verbose=args.verbose, rank=rank)

    # not yet applied on training procedure: channel masking
    channel_masking = ChannelMasking(config).cuda()

    show_message('Initializing optimizer, scheduler and losses...', verbose=args.verbose, rank=rank)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.training_config.lr,
                                 betas=(config.training_config.scheduler_beta_1,
                                        config.training_config.scheduler_beta_2))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training_config.scheduler_step_size,
        gamma=config.training_config.scheduler_gamma
    )
    if config.training_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    show_message('Initializing data loaders...', verbose=args.verbose, rank=rank)
    train_dataset = MelSpecDataset(config, training=True)
    train_sampler = DistributedSampler(train_dataset) if args.n_gpus > 1 else None
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.training_config.batch_size,
        sampler=train_sampler, drop_last=True
    )

    if rank == 0:
        validation_dataset = MelSpecDataset(config, validation=True)
        validation_dataloader = DataLoader(validation_dataset)

    if config.training_config.continue_training:
        show_message('Loading latest checkpoint to continue training...', verbose=args.verbose, rank=rank)
        model, optimizer, iteration = logger.load_latest_checkpoint(model, optimizer)
        epoch_size = len(train_dataset) // config.training_config.batch_size
        epoch_start = iteration // epoch_size
    else:
        iteration = 0
        epoch_start = 0

    if args.n_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        show_message(f'INITIALIZATION IS DONE ON RANK {rank}.')

    show_message('Start training...', verbose=args.verbose, rank=rank)
    try:
        for epoch in range(epoch_start, config.training_config.n_epoch):
            # Training step
            model.train()
            for batch in (
                    tqdm(train_dataloader, leave=False)
                    if args.verbose and rank == 0 else train_dataloader
            ):
                model.zero_grad()

                batch = batch.cuda()
                masked = channel_masking(batch) # random channel is masked

                if config.training_config.use_fp16:
                    with torch.cuda.amp.autocast():
                        loss = (model if args.n_gpus == 1 else model.module).compute_loss(masked, batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss = (model if args.n_gpus == 1 else model.module).compute_loss(masked, batch)
                    loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=config.training_config.grad_clip_threshold
                )

                if config.training_config.use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                loss_stats = {
                    'total_loss': loss.item(),
                    'grad_norm': grad_norm.item()
                }
                logger.log_training(iteration, loss_stats, verbose=False)

                iteration += 1

            # Validation step after epoch on rank==0 GPU
            if epoch % config.training_config.validation_interval == 0 and rank == 0:
                model.eval()
                with torch.no_grad():
                    # Calculate validation set loss
                    validation_loss = 0
                    for i, batch in enumerate(
                        tqdm(validation_dataloader) \
                        if args.verbose and rank == 0 else validation_dataloader
                    ):
                        batch = batch.cuda()    # [1, audio_length]
                        masked = channel_masking(batch)  # random channel is masked

                        validation_loss_ = (model if args.n_gpus == 1 else model.module).compute_loss(masked, batch)
                        validation_loss += validation_loss_
                    validation_loss /= (i + 1)
                    loss_stats = {'total_loss': validation_loss.item()}

                    logger.log_validation(iteration, loss_stats, verbose=args.verbose)

                logger.save_checkpoint(
                    iteration,
                    model if args.n_gpus == 1 else model.module,
                    optimizer
                )
            if epoch % (epoch // 10 + 1) == 0:
                scheduler.step()
    except KeyboardInterrupt:
        print('KeyboardInterrupt: training has been stopped.')
        cleanup()
        return


def run_distributed(fn, config, args):
    try:
        mp.spawn(fn, args=(config, args), nprocs=args.n_gpus, join=True)
    except:
        cleanup()


def init_distributed(rank, n_gpus, dist_config):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    torch.cuda.set_device(rank % n_gpus)

    os.environ['MASTER_ADDR'] = dist_config.MASTER_ADDR
    os.environ['MASTER_PORT'] = dist_config.MASTER_PORT

    torch.distributed.init_process_group(
        backend='nccl', world_size=n_gpus, rank=rank
    )


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='configuration file')
    parser.add_argument(
        '-v', '--verbose', required=False, type=str2bool,
        nargs='?', const=True, default=True, help='verbosity level'
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    n_gpus = torch.cuda.device_count()
    args.__setattr__('n_gpus', n_gpus)

    if args.n_gpus > 1:
        run_distributed(run_training, config, args)
    else:
        run_training(0, config, args)