from data_utils.words import retrieve_word_lists, load_word2vec_model
from data_utils.dataloader import retrieve_data_loader
from models.model import CodenamesModel 
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

def main():
    seed_everything(42)
    noun_path = 'words/filtered_nouns.txt'
    vocab_path = 'words/filtered_vocab.txt'
    word2vec_model = load_word2vec_model('glove-wiki-gigaword-100')
    nouns, vocab = retrieve_word_lists(noun_path, vocab_path)

    board_size = 25
    max_guesses = 10
    model_dim = 128
    num_heads = 8
    num_layers = 6
    lr = 1e-4
    warmup_epochs = 20
    max_epochs = 200
    initial_temperature = 1.0
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
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger('logs', name='codenames')

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        accelerator="gpu",
        devices=[3]
    )
    dataloader = retrieve_data_loader(start_word_index=0, end_word_index=len(nouns), board_size=25, max_guesses=max_guesses, batch_size=32, device='cuda:3')
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()