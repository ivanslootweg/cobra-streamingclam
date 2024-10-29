cd /data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam
# python3 fetch_data.py
FOLD="0" 

python3 main.py \
    --image_path=/data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam/data_source/images \
    --mask_path=/data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam/data_source/tissue_masks \
    --fold="${FOLD}" \
    --train_csv="/data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam/data_source/train.csv" \
    --val_csv="/data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam/data_source/val.csv" \
    --test_csv="/data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam/data_source/test.csv" \
    --mask_suffix="" \
    --mode="fit" \
    --unfreeze_streaming_layers_at_epoch=2 \
    --num_epochs=40 \
    --precision=16 \
    --strategy="ddp_find_unused_parameters_true" \
    --default_save_dir="/data/pathology/projects/DeepDerma/BCC_SCLAM/debug/ckpt" \
    --filetype=".tif" \
    --ckp_path="" \
    --grad_batches=32 \
    --num_gpus=1 \
    --pooling_layer="avgpool" \
    --pooling_kernel=16 \
    --num_classes=2 \
    --learning_rate=1e-4 \
    --read_level=2 \
    --num_workers=3 \
    --branch="sb" \
    --encoder="resnet34" \
    --tile_size_finetune=4800 \
    --tile_size=6400 \
    --image_size=14400 \
    # --tile_size=9600 \ setting this var to 9600 affects execution (termination). default 3200 or 4800 (BCC paper) works 
