import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torch.nn.functional import gumbel_softmax, binary_cross_entropy_with_logits
import wandb
import plotly.graph_objects as go


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
        if self.trainer.is_last_batch:
            self.log_heatmap(clue_weights, "Class Weights")
            self.log_heatmap(num_weights, "Num Weights")

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
        loss = self.compute_loss(output, classes[:, 2:])
        self.log_metrics(output, classes[:, 2:])
        return loss
    
    def compute_loss(self, output, classes):
        return binary_cross_entropy_with_logits(output, classes.float())

    def compute_metrics(self, output, classes):
        probs = F.sigmoid(output)
        predictions = (probs > 0.5) 
        correct = (predictions == classes)
        true_positive = correct * (classes == 1)
        true_negative = correct * (classes == 0)
        accuracy_tp = true_positive.sum() / (classes == 1).sum()
        accuracy_tn = true_negative.sum() / (classes == 0).sum()
        correct_count = (correct*(classes == 1)) / output.shape[0]
        return accuracy_tp.item(), accuracy_tn.item(), correct_count.item()

    def log_metrics(self, predicted, true):
        loss = self.compute_loss(predicted, true).item()
        accuracy_tn, accuracy_tp, correct_count = self.compute_metrics(predicted, true)
        self.log('train_loss', loss)
        self.log('wrong_acc', accuracy_tn)
        self.log('correct_acc', accuracy_tp)
        self.log('correct_count', correct_count)

    def log_heatmap(self, values, labels=None, name='heatmap'):
        if values.is_cuda:
            values = values.cpu()

        values = values.detach().numpy()

        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=[f"Feature {i+1}" for i in range(values.shape[1])],  # Naming the features on x-axis
            y=[f"Batch {i+1}" for i in range(values.shape[0])],    # Naming batches on y-axis
            hoverinfo='x+y+z',  # Showing labels and value on hover
            colorscale='Viridis'
        ))

        fig.update_layout(
            title=f"Heatmap for Batch ID {self.current_epoch}",
            xaxis_title="Features",
            yaxis_title="Batch Instances",
            xaxis={'tickvals': [], 'title': 'Features'},  # No tick labels, but keeps axis title
            yaxis={'tickvals': [], 'title': 'Batch Instances'}  # No tick labels, but keeps axis title
        )

        wandb.log({f"{name}_batch_{self.current_epoch}": wandb.Plotly(fig)})
         
    @torch.no_grad()
    def get_attention_maps(self, words, classes, clue_weights, num_weights):
        attention_maps_clue = self.transformer_clue.get_attention_maps(words)
        attention_maps_guesser = self.transformer_guesser.get_attention_maps(clue_weights, num_weights, words)
        return attention_maps_clue, attention_maps_guesser

    # def on_train_batch_end(self):
    #     lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #     self.log('lr', lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    def log_hp(self):
        self.log('temperature', self.temperature)

    def update_hp(self):
        self.temperature = cosine_scheduler(self.current_epoch, self.trainer.max_epochs, self.hparams.initial_temperature, self.hparams.min_temperature)

    def on_train_epoch_end(self):
        self.update_hp()
        self.log_hp()


