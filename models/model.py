import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.functional import gumbel_softmax, binary_cross_entropy_with_logits

from models.autoencoder import TransformerClue, TransformerGuesser
from models.schedulers import CosineWarmupScheduler, cosine_scheduler

class CodenamesModel(pl.LightningModule):
    def __init__(self, noun_list, vocab_list, word2vec_model, model_dim, max_guess_count, num_heads, num_layers, lr, warmup_epochs, max_epochs, initial_temperature, dropout=0.0, input_dropout=0.0, min_temperature=0.1):
        super().__init__()
        self.hparams['vocab_size'] = len(vocab_list)
        self.save_hyperparameters()

        self.transformer_clue = TransformerClue(
            noun_list=noun_list,
            word2vec_model=word2vec_model,
            model_dim=model_dim,
            vocab_size=len(vocab_list),
            max_guess_count=max_guess_count,
            num_heads=num_heads,
            num_layers=num_layers,
            lr=lr,
            warmup=warmup_epochs,
            initial_temperature=initial_temperature,
            dropout=dropout,
            input_dropout=input_dropout,
            min_temperature=min_temperature
        )

        self.transformer_guesser = TransformerGuesser(
            noun_list=noun_list[2:],
            vocab_list=vocab_list,
            word2vec_model=word2vec_model,
            model_dim=model_dim,
            max_guess_count=max_guess_count,
            num_heads=num_heads,
            num_layers=num_layers,
            lr=lr,
            warmup=warmup_epochs,
            initial_temperature=initial_temperature,
            dropout=dropout,
            input_dropout=input_dropout,
            min_temperature=min_temperature
        )
        self.temperature = initial_temperature

    def forward(self, words, classes):
        word_logits, num_logits = self.transformer_clue(words, classes)
        clue_weights = gumbel_softmax(word_logits, tau=self.temperature, dim=-1)
        num_weights = gumbel_softmax(num_logits, tau=self.temperature, dim=-1)
        output = self.transformer_guesser(clue_weights, num_weights, words)
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup_epochs,
                                             max_epochs=self.hparams.max_epochs,
                                             initial_value=.01,
                                             min_value=.003)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        words, classes = batch
        output = self(words, classes)
        loss = self.compute_loss(output, classes)
        self.log('train_loss', loss)
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        words, classes = batch
        output = self(words, classes)
        loss = self.compute_loss(output, classes)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        words, classes = batch
        output = self(words, classes)
        loss = self.compute_loss(output, classes)
        self.log('test_loss', loss)
        return loss

    def compute_loss(self, output, classes):
        classes = classes[:, 2:].float()
        return binary_cross_entropy_with_logits(output, classes)

    @torch.no_grad()
    def get_attention_maps(self, words, classes, clue_weights, num_weights):
        attention_maps_clue = self.transformer_clue.get_attention_maps(words)
        attention_maps_guesser = self.transformer_guesser.get_attention_maps(clue_weights, num_weights, words)
        return attention_maps_clue, attention_maps_guesser

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_epoch_end(self):
        self.temperature = cosine_scheduler(self.current_epoch, self.trainer.max_epochs, self.hparams.initial_temperature, self.hparams.min_temperature)
