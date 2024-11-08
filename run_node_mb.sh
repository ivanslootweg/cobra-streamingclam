#cd /data/pathology/projects/DeepDerma/BCC_SCLAM/
#python3 fetch_data_blissey.py
FOLD="0"

python3 main.py \
    --image_path=/home/ivanslootweg/data/SBCC/images \
    --mask_path=/home/ivanslootweg/data/SBCC/tissue_masks \
    --fold="${FOLD}" \
    --experiment_name="sclam_cobra_mb_lvl1" \
    --experiment_name="sclam_cobra_mb_lvl1" \
    --train_csv="/home/ivanslootweg/data/SBCC/train.csv" \
    --val_csv="/home/ivanslootweg/data/SBCC/val.csv" \
    --test_csv="/home/ivanslootweg/data/SBCC/test.csv" \
    --mask_suffix="" \
    --mode="fit" \
    --unfreeze_streaming_layers_at_epoch=8 \
    --num_epochs=40 \
    --precision=16 \
    --resume \
    --strategy="ddp_find_unused_parameters_true" \
    --default_save_dir="/home/ivanslootweg/data/SBCC" \
    --filetype=".tif" \
    --ckp_path="" \
    --grad_batches=32 \
    --num_gpus=1 \
    --pooling_layer="avgpool" \
    --pooling_kernel=16 \
    --num_classes=2 \
    --learning_rate=1e-4 \
    --read_level=1\
    --num_workers=3 \
    --branch="mb" \
    --encoder="resnet34" \
    --tile_size_finetune=4800 \
    --tile_size=6400 \
    --image_size=14400 
