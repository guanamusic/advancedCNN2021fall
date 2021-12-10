import os
import argparse
import json
import numpy as np

import torch

from tqdm import tqdm
from datetime import datetime

from data import ChannelMasking, channel_split_n_concat
from model import Inpainter
from utils import ConfigWrapper, show_message, str2bool, parse_filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', required=True,
        type=str, help='configuration file path'
    )
    parser.add_argument(
        '-ch', '--checkpoint_path',
        required=True, type=str, help='checkpoint path'
    )
    parser.add_argument(
        '-md', '--masking_mode',
        required=True, type=str, help='channel masking mode'
    )
    parser.add_argument(
        '-sd', '--save_dir', required=True, type=str, help='directory to save the inpainted spectrogram'
    )
    parser.add_argument(
        '-v', '--verbose', required=False, type=str2bool,
        nargs='?', const=True, default=True, help='verbosity level'
    )
    args = parser.parse_args()

    # Initialize config
    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    if os.path.exists(args.save_dir):
        raise RuntimeError(
            f"You're trying to run inferenece from scratch, "
            f"but directory for inpainted spectrogram `{args.save_dir} already exists. "
            f"Remove it or specify new one.`"
        )
    else:
        os.makedirs(args.save_dir)

    if args.masking_mode == 'True':
        channel_masking = ChannelMasking(config).cuda()
    else:
        raise NotImplementedError()

    # Initialize the model
    model = Inpainter(config)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'], strict=False)

    # Trying to run inference on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Inference
    filelist = os.listdir(config.training_config.test_file_path)
    inference_times = []

    for test_path in (tqdm(filelist, leave=False) if args.verbose else filelist):
        with torch.no_grad():
            model.eval()

            masked = torch.load(config.training_config.test_file_path + test_path)
            masked = masked.to(device).unsqueeze(0)

            if args.masking_mode == 'True':
                masked, mask = channel_masking(masked)
                masked = channel_split_n_concat(config, masked)
                mask = channel_split_n_concat(config, mask)
            else:
                assert 0

            start = datetime.now()
            outputs = model.forward(masked, mask)
            outputs = outputs * mask + masked * (1. - mask)
            end = datetime.now()

            # save it
            outputs = outputs.cpu().squeeze()
            baseidx = os.path.basename(os.path.abspath(test_path)).replace('.pt', '')
            save_path = f'{args.save_dir}/inpainted_{baseidx}.pt'
            torch.save(outputs, save_path)

            inference_time = (end - start).total_seconds()
            inference_times.append(inference_time)

    show_message(f'Done. inference time estimate: {np.mean(inference_times)} ± {np.std(inference_times)}', verbose=args.verbose)