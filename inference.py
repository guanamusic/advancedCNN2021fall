import os
import argparse
import json
import numpy as np

import torch
import torchaudio

from tqdm import tqdm
from datetime import datetime

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
        '-tl', '--test_filelist', required=True, type=str,
        help='masked spectrogram filelist, files of which should be just a torch.Tensor array of shape [6, ?, ?]'
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

    # Initialize the model
    model = Inpainter(config)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'], strict=False)

    # Trying to run inference on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Inference
    filelist = parse_filelist(args.test_filelist)
    inference_times = []

    threshold_str = filelist[0].split('/')[1].split('_')[-1]

    for test_path in (tqdm(filelist, leave=False) if args.verbose else filelist):
        with torch.no_grad():
            model.eval()

            # input shape = [6, ?, ?]
            masked = torch.load(test_path)
            # assert masked.size() == torch.Size([6, ?, ?]), "Input size must be [6, ?, ?]"
            masked = masked.to(device)

            # get some inpainted shit
            start = datetime.now()
            outputs = model.forward(masked)
            end = datetime.now()

            # save it
            outputs = outputs.cpu().squeeze()
            baseidx = os.path.basename(os.path.abspath(test_path)).replace('.pt', '')
            save_path = f'{args.save_dir}/inpainted_{baseidx}.pt'
            torch.save(outputs, save_path)

            inference_time = (end - start).total_seconds()
            inference_times.append(inference_time)

    show_message(f'Done. inference time estimate: {np.mean(inference_times)} ± {np.std(inference_times)}', verbose=args.verbose)