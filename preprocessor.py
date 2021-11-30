import os
import argparse
import json

import numpy as np
import torch
import torchaudio

from data import MelSpectrogramFixed
from utils import ConfigWrapper


def run_preprocessing(config, audio_path, target_path):
    """
    Provides mel-spectrogram dataset for given audio.
    """
    if os.path.exists(target_path):
        raise RuntimeError(
            f"You're trying to run preprocessing from scratch, "
            f"but target directory `{target_path} already exists. Remove it or specify new one.`"
        )
    os.makedirs(target_path)

    audio_length = config.data_config.sample_rate * config.data_config.segment_length_in_ms // 1000
    melSpec = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    )
    n_file = len(sorted(os.listdir(audio_path)))
    assert n_file % config.data_config.n_channel == 0, '# of file is not a multiple of # of channel.'
    filelist = np.reshape(np.array(sorted(os.listdir(audio_path))), (n_file//6, 6))
    for filenames in filelist:
        assert filenames[0][:-8]==filenames[-1][:-8], f'Name sync error!'
        multichannel_audio = []
        for filename in filenames:
            audio, _ = torchaudio.load(audio_path + filename)
            if audio.size()[-1] < audio_length:
                padder = torch.nn.ConstantPad1d((0, audio_length - audio.size()[-1]), 0)
                audio = padder(audio)
            multichannel_audio.append(audio.squeeze()[:audio_length])
        multichannel_audio = torch.stack((multichannel_audio))
        # [6, 128, 390]
        multichannel_melspec = melSpec(multichannel_audio)
        torch.save(multichannel_melspec, target_path + filenames[0][:-8] + '.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='configuration file')
    parser.add_argument('-ad', '--audio_dir', required=True, type=str, help='audio file directory')
    parser.add_argument('-td', '--target_dir', required=True, type=str, help='mel spectrogram save directory')
    args = parser.parse_args()
    audio_path = args.audio_dir
    target_path = args.target_dir

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    n_gpus = torch.cuda.device_count()
    assert n_gpus == 1, 'Only one GPU is allowed.'
    args.__setattr__('n_gpus', n_gpus)

    run_preprocessing(config=config, audio_path=audio_path, target_path=target_path)