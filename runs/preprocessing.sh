export CUDA_VISIBLE_DEVICES=1

CONFIG_PATH=configs/default.json
AUDIO_DIR=chime_3_simu_only/
TARGET_DIR=chime_3_simu_only_melspec/

python preprocessor.py -c $CONFIG_PATH -ad $AUDIO_DIR -td $TARGET_DIR
