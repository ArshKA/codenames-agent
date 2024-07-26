from data_utils.words import retrieve_word_lists, load_word2vec_model
from data_utils.dataloader import retrieve_data_loader
from models.model import CodenamesModel 
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb

def main():
    seed_everything(42)
    wandb.require("core")


    noun_path = 'words/filtered_nouns.txt'
    vocab_path = 'words/filtered_vocab.txt'
    word2vec_model = load_word2vec_model('glove-wiki-gigaword-100')
    nouns, vocab = retrieve_word_lists(noun_path, vocab_path)

    board_size = 25
    max_guesses = 25
    model_dim = 128
    num_heads = 8
    num_layers = 6
    lr = 1e-4
    warmup_epochs = 20
    max_epochs = 200
    initial_temperature = 100.0
    dropout = 0.1
    input_dropout = 0.1
    min_temperature = 0.1

    model = CodenamesModel(
        noun_list=nouns,
        vocab_list=vocab,
        word2vec_model=word2vec_model,
        model_dim=model_dim,
        max_guess_count=max_guesses,
        num_heads=num_heads,
        num_layers=num_layers,
        lr=lr,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        initial_temperature=initial_temperature,
        dropout=dropout,
        input_dropout=input_dropout,
        min_temperature=min_temperature
    )

    dataloader = retrieve_data_loader(start_word_index=0, end_word_index=len(nouns), board_size=board_size, max_guesses=max_guesses, batch_size=32, device='cpu')


    # for words, classes in dataloader:
    #     print(words.shape)
    #     print(classes.shape)
    #     out = model.training_step((words, classes), 1)
    #     print(out.shape)
    #     break

    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(project="Codenames")
    wandb_logger.watch(model, log="all", log_graph=True, log_freq=20)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        # accelerator="cpu",
        # devices=[0]
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()