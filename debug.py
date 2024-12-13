from main import *
import time 
import os
import shutil
import glob

OUT_PATH = Path("/data/temporary/ivan/DeepDerma/BCC_SCLAM/embeddings")
OUT_LOCAL = Path("/home/embeddings")

def write_embedding(image_name, embedding,mask,width):
    cache_path = OUT_LOCAL / f"{image_name}.pt"
    if not os.path.exists(cache_path):     
        _out = {"embedding": embedding.squeeze(0),
            "mask": mask,
                "width": width
            }
 
        torch.save(_out, cache_path) 
        cache_path = OUT_PATH  / f"{image_name}.pt"
        torch.save(_out, cache_path) 
        del _out


def move_unique_files(src_dir = Path(""), dst_dir = Path("")):
    src_files = os.listdir(src_dir)
    dst_files = os.listdir(dst_dir)
    new_files = src_files - dst_files
    for file_path in new_files:
        shutil.copyfile(src_dir / file_path, dst_dir / file_path)

    """alternatively ( not recommended ) : move the full directory """
    # copytree(str(self.embeddings_temp_dir),str(self.embeddings_source),dirs_exist_ok=True)


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
            idx = list(dataloader.sampler)[_step]
            if _step < 3315:
                continue
            try:
                batch = dataloader.dataset.__getitem__(idx)
                write_embedding(batch["image_name"],batch["image"],batch["mask"], batch["width"])
            except Exception as e:
                print("error: ", batch["image_name"], e)
                continue
    
    move_unique_files(options.embeddings_temp_dir, options.embeddings_source)
