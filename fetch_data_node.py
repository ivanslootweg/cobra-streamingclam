import os
import pandas as pd
from pathlib import Path
import numpy as np

DATASET_SIZE = None
DATA_OUT = Path("/home/ivanslootweg/data/SBCC")

def symlink_force(src, dest):
# https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python
    try:
        os.symlink(src, dest)
    except FileExistsError as e:
        return

def create_symlinks(label_csv,split="train"):
    label_csv["slide"] = list(map(lambda x: f'{split}_' + x, map(str,np.arange(len(label_csv)))))

    for (slide,uuid) in label_csv[["slide","uuid"]].itertuples(index=False):
        image_source_path = data_source_folder / f"images/{uuid}.tif"
        mask_source_path = data_source_folder / f"tissue_masks/{uuid}.tif"
        symlink_force(image_source_path,DATA_OUT / f"images/{slide}.tif")
        symlink_force(mask_source_path,DATA_OUT / f"tissue_masks/{slide}.tif")
        
    label_csv.to_csv(DATA_OUT / f"{split}_keys.csv",index=False)
    label_csv = label_csv[["slide","tumor"]]
    label_csv.to_csv(DATA_OUT / f"{split}.csv",index=False)


if __name__ == "__main__":
    if not os.path.exists(DATA_OUT / "images"):
        os.makedirs(DATA_OUT / "images")

    if not os.path.exists(DATA_OUT / "tissue_masks"):
        os.makedirs(DATA_OUT / "tissue_masks")

    data_source_folder = Path("/data/pa_cpgarchive/archives/skin/cobra/data/biopsies")
    label_folder = Path("/data/pa_cpgarchive/archives/skin/cobra/folds/bcc_risk")

    for split in ("train","test","val"):
        label_path = label_folder / f"{split}.csv"
        if DATASET_SIZE:
            labels = pd.read_csv(label_path)[:DATASET_SIZE] 
        else:
            labels = pd.read_csv(label_path)
        create_symlinks(labels,split=split)    
