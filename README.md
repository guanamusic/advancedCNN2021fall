# ADVANCED CNN 2021 FALL

## Introduction
In a multi-channel recording environment, if one or more channels of the multichannel microphone are hindered by damage such as physical impact, the spatial representation of the recorded audio is broken.

Hence, we solve the task of inpainting a single damaged channel of the microphone array through the deep neural network to resolve the issue.

![task_brief](https://user-images.githubusercontent.com/52475881/146413612-80cc0c07-9403-454f-90e8-0d5f8313f6ee.png)


## Requirements
    torch == 1.6.0
    torchaudio == 0.6.0
    numpy == 1.18.5
    matplotlib >= 3.3.1
    tqdm
    tensorboard


## Getting started
- Preprocessing (to make mel-spectrogram from audio):
```
sh runs/preprocessing.sh
```
- Train:
```
sh runs/train.sh
```
- Inference:
```
sh runs/inference.sh
```


## Model architecture
- Generator
    - Generative model with contextual attention from [Tensorflow implementation](https://github.com/JiahuiYu/generative_inpainting), and [Pytorch implementation](https://github.com/daa233/generative-inpainting-pytorch) was used as the base architecture.

    - The generator is constructed with a coarse network followed by a refinement network.

        ![AdvCNN_Gen](https://user-images.githubusercontent.com/52475881/146397307-b15721cd-7212-4d85-9614-b7bbcd1f777c.png)
    
- Discriminator
    - The local discriminator discriminates on the inpainted result of the masked channel only.

    - The global discriminator discriminates on the output with all six channels.

        ![AdvCNN_Disc](https://user-images.githubusercontent.com/52475881/146396722-f62acb59-2246-489b-9d15-474cb7647711.png)




## Inpainting Result
- Original Mel-spectrogram

    ![F01_BUS_original](https://user-images.githubusercontent.com/52475881/146398242-8ce1c294-e1ce-4b07-9fda-448d8764837d.png)

- Inpainted Mel-spectrogram

    ![F01_BUS_inpainted](https://user-images.githubusercontent.com/52475881/146398161-126a7597-b6d8-4862-addd-33a1d963fb19.png)