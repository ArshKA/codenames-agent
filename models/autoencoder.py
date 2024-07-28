import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torch.nn.functional import gumbel_softmax

from models.embedding import EncoderEmbedding, DecoderEmbedding
from models.transformer import TransformerEncoder


class TransformerClue(pl.LightningModule):
    def __init__(self, noun_list, word2vec_model, model_dim, vocab_size, max_guess_count, num_heads, num_layers, lr, warmup, initial_temperature, dropout=0.0, input_dropout=0.0, min_temperature=0.1):
        super().__init__()
        self.hparams['embedding_dim'] = word2vec_model.vector_size
        self.hparams['noun_size'] = len(noun_list)
        self.save_hyperparameters('model_dim', 'vocab_size', 'max_guess_count', 'num_heads', 'num_layers', 'lr', 'warmup', 'initial_temperature', 'dropout', 'input_dropout', 'min_temperature')

        self._initialize_embeddings(noun_list, word2vec_model)
        self._create_model()

    def _initialize_embeddings(self, noun_list, word2vec_model):
        self.embedding_model = EncoderEmbedding(noun_list, word2vec_model)

    def _create_model(self):
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.embedding_dim, self.hparams.model_dim)
        )
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        self.vocab_out = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.embedding_dim)
        )

    def forward(self, words, classes):
        x = self.embedding_model(words, classes)
        x = self.input_net(x)
        x = self.transformer(x)
        word_logits = self.vocab_out(x[:, 0])
        return word_logits

    @torch.no_grad()
    def get_attention_maps(self, x):
        x = self.input_net(x)
        x = self.embedding_model(x)
        attention_maps = self.transformer.get_attention_maps(x)
        return attention_maps


class TransformerGuesser(pl.LightningModule):
    def __init__(self, noun_list, vocab_list, word2vec_model, model_dim, max_guess_count, num_heads, num_layers, lr, warmup, initial_temperature, dropout=0.0, input_dropout=0.0, min_temperature=0.1):
        super().__init__()
        self.hparams['embedding_dim'] = word2vec_model.vector_size
        self.hparams['noun_size'] = len(noun_list)
        self.hparams['vocab_size'] = len(vocab_list)

        self.save_hyperparameters('model_dim', 'max_guess_count', 'num_heads', 'num_layers', 'lr', 'warmup', 'initial_temperature', 'dropout', 'input_dropout', 'min_temperature')

        self._initialize_embeddings(noun_list, vocab_list, word2vec_model)
        self._create_model()
        self.temperature = initial_temperature

    def _initialize_embeddings(self, noun_list, vocab_list, word2vec_model):
        self.embedding_model = DecoderEmbedding(noun_list, vocab_list, word2vec_model, self.hparams.max_guess_count)

    def _create_model(self):
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.embedding_dim, self.hparams.model_dim)
        )
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, 1)
        )

    def forward(self, clue_weights, words):
        x = self.embedding_model(clue_weights, words)
        x = self.input_net(x)
        x = self.transformer(x)
        x = self.output_net(x[:, 1:])
        return torch.squeeze(x)

    @torch.no_grad()
    def get_attention_maps(self, clue_weights, words):
        x = self.embedding_model(clue_weights, words)
        x = self.input_net(x)
        attention_maps = self.transformer.get_attention_maps(x)
        return attention_maps
