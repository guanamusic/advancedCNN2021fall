export CUDA_VISIBLE_DEVICES=4

CONFIG_PATH=configs/default.json
CHECKPOINT_PATH=logs/211207_try_channelmasking_modified/checkpoint_199.pt
CHANNEL_MASKING_MODE=True
SAVE_DIR=generated/default
VERBOSE='yes'

python inference.py -c $CONFIG_PATH -ch $CHECKPOINT_PATH -md $CHANNEL_MASKING_MODE -sd $SAVE_DIR -v $VERBOSE
