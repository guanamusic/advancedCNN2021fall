import os
import numpy as np

import torch
from torchaudio.transforms import MelSpectrogram

np.random.seed(1234)
torch.manual_seed(1234)


class MelSpecDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """
    def __init__(self, config, training=False, validation=False):
        super(MelSpecDataset, self).__init__()
        self.config = config
        self.training = training
        self.validation = validation

        assert not (self.training and self.validation), \
            "Check whether the dataloader mode is training or validation."

        if self.training:
            self.file_path = config.training_config.train_file_path
        elif self.validation:
            self.file_path = config.training_config.validation_file_path
        else:
            self.file_path = config.training_config.test_file_path

        self.filenames = os.listdir(self.file_path)

    def __getitem__(self, index):
        mel_spec = torch.load(self.file_path + self.filenames[index])
        return mel_spec

    def __len__(self):
        return len(self.filenames)

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

        assert channel_mask.size() == batch.size(), "WHAT ON EARTH IS GOING ON??????"
        output = batch * (torch.ones_like(channel_mask) - channel_mask)   # masking

        return output, channel_mask.type(torch.FloatTensor).to(batch.device)


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


def channel_split_n_concat(config, multi_channel_mel_spec):
    assert len(multi_channel_mel_spec.size()) == 4
    n_width = config.data_config.channel_concat_width
    n_height = config.data_config.channel_concat_height
    assert n_height * n_width == config.data_config.n_channel

    mel_specs = torch.split(multi_channel_mel_spec, 1, dim=1)
    mel_specs_concat = torch.cat(tuple(torch.cat(tuple(mel_specs[idx + n_width * jdx] for idx in range(n_width)), dim=3) for jdx in range(n_height)), dim=2)

    assert mel_specs_concat.device == multi_channel_mel_spec.device

    return mel_specs_concat
