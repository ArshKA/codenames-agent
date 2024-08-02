import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class EncoderEmbedding(nn.Module):
    def __init__(self, noun_list, word2vec_model):
        super(EncoderEmbedding, self).__init__()
        self.word2vec_model = word2vec_model
        self.embedding_dim = self.word2vec_model.vector_size
        self.noun_embedding = nn.Embedding.from_pretrained(self._create_embedding_matrix(noun_list), freeze=True)
        self.class_embedding = nn.Embedding(2, self.embedding_dim)
        self.special_embeddings = nn.Embedding(1, self.embedding_dim)

        self.positional_encoding = PositionalEncoding(self.embedding_dim, max_len=25)
        # torch.nn.init.zeros_(self.class_embedding.weight)


    def _create_embedding_matrix(self, word_list):

        embedding_matrix = torch.zeros(len(word_list), self.embedding_dim)

        for idx, word in enumerate(word_list):
            if word in self.word2vec_model:
                embedding_matrix[idx] = torch.tensor(self.word2vec_model[word])
            else:
                raise Exception(f"Word '{word}' not found in the Word2Vec model.")

        return embedding_matrix

    def forward(self, words, classes):
        out = torch.zeros((words.shape[0], words.shape[1]+1, self.embedding_dim), device=words.device)
        out[:, 0] = self.special_embeddings.weight[0]
        out[:, 1:] = self.noun_embedding(words) + self.class_embedding(classes)
        out[:, 1:] = self.positional_encoding(out[:, 1:])
        return out
    

class DecoderEmbedding(nn.Module):
    def __init__(self, noun_list, vocab_list, word2vec_model, max_num):
        super(DecoderEmbedding, self).__init__()
        self.word2vec_model = word2vec_model
        self.embedding_dim = self.word2vec_model.vector_size
        self.noun_embedding = nn.Embedding.from_pretrained(self._create_embedding_matrix(noun_list), freeze=True)
        # self.vocab_embedding = nn.Embedding.from_pretrained(self._create_embedding_matrix(vocab_list), freeze=True)
        self.special_embeddings = nn.Embedding(1, self.embedding_dim)
        # torch.nn.init.normal_(self.special_embeddings.weight, 0, .05)

        self.positional_encoding = PositionalEncoding(self.embedding_dim, max_len=25)


    def _create_embedding_matrix(self, word_list):

        embedding_matrix = torch.zeros(len(word_list), self.embedding_dim)

        for idx, word in enumerate(word_list):
            if word in self.word2vec_model:
                embedding_matrix[idx] = torch.tensor(self.word2vec_model[word])
            else:
                raise Exception(f"Word '{word}' not found in the Word2Vec model.")

        return embedding_matrix

    def forward(self, clue_weights, words):
        out = torch.zeros((words.shape[0], words.shape[1]+1, self.embedding_dim), device=words.device)
        out[:, 0] = clue_weights + self.special_embeddings.weight[0]
        out[:, 1:] = self.noun_embedding(words)
        out[:, 1:] = self.positional_encoding(out[:, 1:])
        return out