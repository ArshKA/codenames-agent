import torch
from torch import nn

class EncoderEmbedding(nn.Module):
    def __init__(self, word_list, word2vec_model):
        super(EncoderEmbedding, self).__init__()
        self.word2vec_model = word2vec_model
        self.embedding_dim = self.word2vec_model.vector_size
        self.noun_embedding = nn.Embedding.from_pretrained(self._create_embedding_matrix(word_list), freeze=True)
        self.class_embedding = nn.Embedding(2, self.embedding_dim)
        self.special_embeddings = nn.Embedding(1, self.embedding_dim)
        # torch.nn.init.normal_(self.class_embedding.weight, 0, .1)


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
        return out
    

class DecoderEmbedding(nn.Module):
    def __init__(self, noun_list, vocab_list, word2vec_model, max_num):
        super(DecoderEmbedding, self).__init__()
        self.word2vec_model = word2vec_model
        self.embedding_dim = self.word2vec_model.vector_size
        self.noun_embedding = nn.Embedding.from_pretrained(self._create_embedding_matrix(noun_list), freeze=True)
        self.vocab_embedding = nn.Embedding.from_pretrained(self._create_embedding_matrix(vocab_list), freeze=True)
        self.special_embeddings = nn.Embedding(1, self.embedding_dim)
        # torch.nn.init.normal_(self.special_embeddings.weight, 0, .05)


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
        return out