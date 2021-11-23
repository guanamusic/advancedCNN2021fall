export CUDA_VISIBLE_DEVICES=4

CONFIG_PATH=configs/default.json
CHECKPOINT_PATH=logs/checkpoint_somekindofshit.pt
TEST_FILELIST_PATH=filelists/test_file_list_or_some_kind_of_stuff.txt
SAVE_DIR=generated/default
VERBOSE='yes'

python inference.py -c $CONFIG_PATH -ch $CHECKPOINT_PATH -tl $TEST_FILELIST_PATH -sd $SAVE_DIR -v $VERBOSE
