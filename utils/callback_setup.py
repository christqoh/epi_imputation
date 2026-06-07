from typing import List

from pytorch_lightning.callbacks import ModelCheckpoint


def return_callbacks(model_dir) -> List:
    callbacks = []
    best_val_loss_checkpoint_callback = ModelCheckpoint(dirpath=model_dir, every_n_epochs=1, save_top_k=1,
                                                        monitor='val/loss', mode='min', filename='best_val_loss',
                                                        save_on_train_epoch_end=True)
    callbacks.append(best_val_loss_checkpoint_callback)

    every_n_model_checkpoint_callback = ModelCheckpoint(dirpath=model_dir, every_n_epochs=50, save_top_k=1,
                                                        save_last=True, filename='last')
    callbacks.append(every_n_model_checkpoint_callback)

    return callbacks
