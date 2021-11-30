import numpy as np

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

np.random.seed(1234)
torch.manual_seed(1234)

from utils import parse_filelist


class AudioDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """
    def __init__(self, config, training=True, validation=True):
        super(AudioDataset, self).__init__()
        self.config = config
        self.training = training
        self.validation = validation

        self.segment_length = config.data_config.segment_length
        self.sample_rate = config.data_config.sample_rate

        self.filelist_path = config.training_config.train_filelist_path \
            if self.training else config.training_config.test_filelist_path
        self.audio_paths = parse_filelist(self.filelist_path)

    def load_audio_to_torch(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)
        # To ensure to create integer # segments will be processed in a right way for full signals
        if not (self.training or self.validation):
            p = (audio.shape[-1] // self.segment_length + 1) * self.segment_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
        return audio.squeeze(), sample_rate

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        audio, sample_rate = self.load_audio_to_torch(audio_path)

        assert sample_rate == self.sample_rate, \
            f"""Got path to audio of sampling rate {sample_rate}, \
                but required {self.sample_rate} according config."""

        if not (self.training or self.validation):  # If test
            return audio
        # Take segment of audio for training
        if audio.shape[-1] > self.segment_length:
            max_audio_start = audio.shape[-1] - self.segment_length
            audio_start = np.random.randint(0, max_audio_start)
            segment = audio[audio_start:audio_start+self.segment_length]
        else:
            segment = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.shape[-1]), 'constant'
            ).data
        return segment

    def __len__(self):
        return len(self.audio_paths)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class ChannelMasking(torch.nn.Module):
    """
    For masking an arbitrary single channel
    """
    def __init__(self, config):
        super(ChannelMasking, self).__init__()
        self.n_channel = config.data_config.n_channel

    def forward(self, batch):
        # batch.size() = torch.size([batch_size, 6(channel_size), ?, ?])
        assert len(batch.size()) == 4, "Input is not a batch. Check the input type."
        assert batch.size()[1] == self.n_channel, f"""# of channel is {batch.size()[1]}, 
                                                    but required {self.n_channel} according config."""

        masked_channel = torch.randint(low=0, high=self.n_channel, size=(batch.size()[0], )).cuda()
        channel_mask = torch.nn.functional.one_hot(masked_channel,
                                                   num_classes=self.n_channel).unsqueeze(2).unsqueeze(3)
        channel_mask = channel_mask.expand_as(batch)
        channel_mask = torch.ones_like(channel_mask) - channel_mask

        assert channel_mask.size() == batch.size(), "WHAT ON EARTH IS GOING ON??????"
        output = batch * channel_mask   # masking

        return output


class MelSpectrogramFixed(torch.nn.Module):
    """
    make some mel_spec shit
    """
    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = 20 * self.torchaudio_backend(x).log10()
        # Clipping mel-spec value on [-70, 30]
        outputs[outputs < -70] = -70
        outputs[outputs > 30] = 30
        return outputs
