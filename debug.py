from main import *
import time 
import os

if __name__ == "__main__":
    pl.seed_everything(1)

    options = get_options()
    streaming_options = get_streaming_options(options)

    model = configure_streamingclam(options, streaming_options)
    tile_stride, network_output_stride = get_model_statistics(model)
    options.tile_stride = tile_stride

    if options.stream_pooling_kernel:
        options.network_output_stride = network_output_stride
    else:
        options.network_output_stride = max(network_output_stride * options.pooling_kernel, network_output_stride)
    dm = configure_datamodule(options)
    dm.setup(stage=options.mode)

    if options.mode == "fit":
        dataloader = dm.train_dataloader()
        total = len(dataloader.dataset)
        for _step in tqdm(range(total),total=total):
            if  (1537 >= _step) and (_step >= 1534):
                idx = list(dataloader.sampler)[_step]
                print(_step,idx)
                batch = dataloader.dataset.__getitem__(idx)
                model.training_step(batch,idx)
            else:
                time.sleep(0.001)
                continue
