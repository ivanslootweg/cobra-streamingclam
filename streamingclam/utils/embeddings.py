import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import glob
from typing import Callable
import logging
import os
from pathlib import Path
import shutil

log = logging.getLogger(__name__)


def multiplicative(epoch: int) -> float:
    return 2.0


class IntermediateEmbeddings(BasePredictionWriter):
    def __init__(
        self,
        embeddings_source: Path,
        use_embeddings: bool = False,
        unfreeze_at_epoch : int = 10,
        embeddings_temp_dir : Path = Path("/home/embeddings"),
        export_to_remote_every : int = 50
    ):
        super().__init__()
        self.embeddings_source = embeddings_source
        self.use_embeddings = use_embeddings
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.embeddings_temp_dir = embeddings_temp_dir
        self.new_local_embedding = False # switch to know whether to export embeddings to remote or not
        self.export_to_remote_every = export_to_remote_every
        self.save_embeddings = False
        self.train_batch_size = None
        self.val_batch_size = None

    def write_on_batch_end(self,**kwargs):
        pass

    def write_on_epoch_end(self,**kwargs):
        pass

    def on_val_batch_end(self,**kwargs):
        pass

    def on_predict_batch_end(self,**kwargs):
        pass
   
    def on_predict_epoch_end(self,**kwargs):
        pass

    @rank_zero_only
    def move_unique_files(self,src_dir = Path(""), dst_dir = Path("")):
        src_files = os.listdir(src_dir)
        dst_files = os.listdir(dst_dir)
        new_files = src_files - dst_files
        for file_path in new_files:
            shutil.copyfile(src_dir / file_path, dst_dir / file_path)

        """alternatively ( not recommended ) : move the full directory """
         # copytree(str(self.embeddings_temp_dir),str(self.embeddings_source),dirs_exist_ok=True)
 
    def write_embedding(self,image_name, embedding,mask,width,batch_idx):
        _out = {"embedding": embedding.detach().cpu().squeeze(0),
            "mask": mask,
            "width": width
        }
        cache_path = self.embeddings_temp_dir /  f"{image_name}.pt"
        torch.save(_out, cache_path) 
        cache_path = self.embeddings_source /  f"{image_name}.pt"
        torch.save(_out, cache_path) 
        self.new_local_embedding = True
        del _out

    def on_train_start(self,trainer,pl_module):
        # when encoder is frozen and intermediate embeddings are used 
        if (trainer.current_epoch < (self.unfreeze_at_epoch - 1)) and self.use_embeddings:           
            # copy existing embeddings from remote
            print("============== start using intermediate embeddings ================" )
            self.train_batch_size =  trainer.datamodule.train_dataloader().batch_size
            self.val_batch_size =  trainer.datamodule.val_dataloader().batch_size
            
            os.makedirs(self.embeddings_temp_dir, exist_ok=True)
            """optionally move embeddings from remove to local at the start of training. for now we decice to make the user responsible
            to move files from source to temp """
            # copytree(str(self.embeddings_source), str(self.embeddings_temp_dir),dirs_exist_ok=True)

            # instruct dataloader to load any precomputed embeddings
            trainer.datamodule.load_embeddings = self.use_embeddings
            trainer.datamodule.embeddings_source = self.embeddings_temp_dir
            trainer.datamodule.reset("fit")

            # instruct to save embeddings 
            self.save_embeddings = self.use_embeddings

    def save_batch(self,pl_module,embeddings,batch_idx,image_name,batch_size):
        # save embeddings locally
        if self.save_embeddings and pl_module.embedding_computed:
            for _item in range(batch_size):
                embedding = embeddings[_item].unsqueeze(0)
                if batch_size > 1 :
                    image_name = image_name[_item]
                    mask = pl_module.mask[_item]
                    width = pl_module.width[_item]
                else:
                    image_name = image_name
                    mask = pl_module.mask
                    width = pl_module.width

                self.write_embedding(image_name,embedding,mask,width, batch_idx)

        if self.new_local_embedding and ((batch_idx+1) % self.export_to_remote_every) == 0:
            if not self.embeddings_temp_dir == self.embeddings_source:
                """ for now we decide to save on remove on every save operation in save_batch. alternatively, move_unique_files all at once """
                # self.move_unique_files(self.embeddings_temp_dir,self.embeddings_source)
                self.new_local_embedding = False

    def on_train_batch_end(self,trainer, pl_module, output, batch, batch_idx):
        image_name = batch["image_name"]
        self.save_batch(pl_module,pl_module.str_output,batch_idx,image_name,self.train_batch_size)
        del pl_module.str_output, pl_module.image

    def on_validation_batch_end(self,trainer, pl_module, output, batch, batch_idx):
        image_name = batch["image_name"]    
        self.save_batch(pl_module,pl_module.str_output,batch_idx,image_name,self.val_batch_size)
        del pl_module.str_output, pl_module.image

    # def on_epoch_end(self, trainer, pl_module,predictions,batch_indices):
    def on_validation_end(self,trainer, pl_module):
        # backup embeddings on remote
        if self.use_embeddings and self.new_local_embedding:
            if not self.embeddings_temp_dir == self.embeddings_source:
                """ for now we decide to save on remove on every save operation in save_batch. alternatively, move_unique_files all at once """
                # self.move_unique_files(self.embeddings_temp_dir,self.embeddings_source)
                self.new_local_embedding = False


        # stop using intermediate embeddings when unfreezing the encoder
        if trainer.current_epoch == (self.unfreeze_at_epoch-1):
            print("============== stop using intermediate embeddings ================" )
            self.use_embeddings = False
            self.save_embeddings = False
            self.new_local_embedding = False

            trainer.datamodule.load_embeddings = False
            trainer.datamodule.verbose = False
            trainer.datamodule.reset("fit")  


 

