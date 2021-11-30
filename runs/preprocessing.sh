export CUDA_VISIBLE_DEVICES=1

CONFIG_PATH=configs/default.json
AUDIO_DIR=chime_3_simu_only/et05_str_simu/
TARGET_DIR=chime_3_simu_only/et05_str_simu_melspec/

python preprocessor.py -c $CONFIG_PATH -ad $AUDIO_DIR -td $TARGET_DIR
