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
from data import MelSpecDataset, ChannelMasking, channel_split_n_concat
from utils import ConfigWrapper, show_message, str2bool

from tools import spatial_discounting_mask


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
    optimizer_g = torch.optim.Adam(params=model.netG.parameters(),
                                   lr=config.training_config.lr,
                                   betas=(config.training_config.scheduler_beta_1,
                                          config.training_config.scheduler_beta_2))

    optimizer_d = torch.optim.Adam(params=list(model.localD.parameters())+list(model.globalD.parameters()),
                                   lr=config.training_config.lr,
                                   betas=(config.training_config.scheduler_beta_1,
                                          config.training_config.scheduler_beta_2))
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optimizer_g,
        step_size=config.training_config.scheduler_step_size,
        gamma=config.training_config.scheduler_gamma
    )
    scheduler_d = torch.optim.lr_scheduler.StepLR(
        optimizer_d,
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
        model, optimizer_g, optimizer_d, epoch = logger.load_latest_checkpoint(model, optimizer_g, optimizer_d)
        epoch_size = len(train_dataset) // config.training_config.batch_size
        iteration = (epoch + 1) * epoch_size
        epoch_start = epoch + 1
    else:
        iteration = 0
        epoch_start = 0

    if args.n_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        show_message(f'INITIALIZATION IS DONE ON RANK {rank}.')

    assert len(train_dataset.__getitem__(0).size()) == 3, f"{train_dataset.__getitem__(0).size()}"
    _, width, height = train_dataset.__getitem__(0).size()
    sd_mask = spatial_discounting_mask(config, width=width, height=height)

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
                masked_batch, mask = channel_masking(batch) # random channel is masked
                sd_mask = sd_mask.to(batch.device)

                batch = channel_split_n_concat(config, batch)
                masked_batch = channel_split_n_concat(config, masked_batch)
                mask = channel_split_n_concat(config, mask)

                compute_g_loss = iteration % config.training_config.n_critic == 0
                if config.training_config.use_fp16:
                    with torch.cuda.amp.autocast():
                        loss = (model if args.n_gpus == 1
                                else model.module).compute_loss(masked_input=masked_batch,
                                                                mask=mask,
                                                                ground_truth=batch,
                                                                spatial_discounting_mask=sd_mask,
                                                                compute_g_loss=compute_g_loss)
                    scaler.scale(loss['total_discriminator']).backward()
                    scaler.unscale_(optimizer_d)
                    if compute_g_loss:
                        scaler.scale(loss['total_generator']).backward()
                        scaler.unscale_(optimizer_g)
                else:
                    loss = (model if args.n_gpus == 1
                            else model.module).compute_loss(masked_input=masked_batch,
                                                            mask=mask,
                                                            ground_truth=batch,
                                                            spatial_discounting_mask=sd_mask,
                                                            compute_g_loss=compute_g_loss)
                    loss['total_discriminator'].backward()
                    if compute_g_loss:
                        loss['total_generator'].backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=config.training_config.grad_clip_threshold
                )

                if config.training_config.use_fp16:
                    scaler.step(optimizer_d)
                    if compute_g_loss:
                        scaler.step(optimizer_g)
                    scaler.update()
                else:
                    optimizer_d.step()
                    if compute_g_loss:
                        optimizer_g.step()

                if not compute_g_loss:
                    loss_stats = {
                        'total_D_loss': loss['total_discriminator'].item(),
                        'grad_norm': grad_norm.item()
                    }
                else:
                    loss_stats = {
                        'total_D_loss': loss['total_discriminator'].item(),
                        'total_G_loss': loss['total_generator'].item(),
                        'grad_norm': grad_norm.item()
                    }
                logger.log_training(iteration, loss_stats, verbose=False)

                iteration += 1

            # Validation step after epoch on rank==0 GPU
            if epoch % config.training_config.validation_interval == 0 and rank == 0:
                model.eval()
                with torch.no_grad():
                    # Calculate validation set loss
                    validation_loss_d = 0
                    validation_loss_g = 0
                    for i, batch in enumerate(
                        tqdm(validation_dataloader) \
                        if args.verbose and rank == 0 else validation_dataloader
                    ):
                        batch = batch.cuda()    # [1, audio_length]
                        masked_batch, mask = channel_masking(batch)  # random channel is masked
                        sd_mask = sd_mask.to(batch.device)

                        batch = channel_split_n_concat(config, batch)
                        masked_batch = channel_split_n_concat(config, masked_batch)
                        mask = channel_split_n_concat(config, mask)

                        validation_loss_ = (model if args.n_gpus == 1
                                            else model.module).compute_loss(masked_input=masked_batch,
                                                                            mask=mask,
                                                                            ground_truth=batch,
                                                                            spatial_discounting_mask=sd_mask,
                                                                            compute_g_loss=True,
                                                                            compute_d_loss=False)
                        # validation_loss_d += validation_loss_['total_discriminator']
                        validation_loss_g += validation_loss_['total_generator']
                    # validation_loss_d /= (i + 1)
                    validation_loss_g /= (i + 1)
                    loss_stats = {
                        # 'total_D_loss': validation_loss_d.item(),
                        'total_G_loss': validation_loss_g.item()
                    }

                    logger.log_validation(epoch, loss_stats, verbose=args.verbose)

                logger.save_checkpoint(
                    epoch,
                    model if args.n_gpus == 1 else model.module,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d
                )
            if epoch % (epoch // 10 + 1) == 0:
                scheduler_d.step()
                scheduler_g.step()
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