
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb


from data_utils.words import retrieve_word_lists, load_word2vec_model
from data_utils.dataloader import retrieve_data_loader
from models.model import CodenamesModel 
from helpers import retrieve_hparams

CONFIG_PATH = 'hyperparameters.yaml'

def main():
    seed_everything(42)

    wandb.require("core")

    noun_path = 'words/filtered_nouns.txt'
    vocab_path = 'words/filtered_vocab.txt'
    word2vec_model = load_word2vec_model('glove-wiki-gigaword-100')
    nouns, vocab = retrieve_word_lists(noun_path, vocab_path)
    nouns = nouns[:100]
    vocab = nouns[2:]

    hparams = retrieve_hparams(CONFIG_PATH)

    model = CodenamesModel(
        noun_list=nouns,
        vocab_list=vocab,
        word2vec_model=word2vec_model,
        **hparams
    )

    dataloader = retrieve_data_loader(
        start_word_index=2,
        end_word_index=len(nouns),
        board_size=25,  # Keeping the board_size as it was not part of the hparams
        max_guesses=hparams['max_guess_count'],
        batch_size=32,
        num_workers=2,
        device='cpu'
    )

    # ModelCheckpoint callback to save the best models
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    # LearningRateMonitor callback to log the learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="Codenames")
    wandb_logger.watch(model, log="all", log_graph=True, log_freq=5000)

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=100,
        accelerator="cpu",
        # devices=[3]
    )

    # Train the model
    trainer.fit(model, dataloader)



if __name__ == '__main__':
    main()