import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torch.nn.functional import gumbel_softmax, binary_cross_entropy_with_logits
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from models.autoencoder import TransformerClue, TransformerGuesser
from models.schedulers import CosineWarmupScheduler, cosine_scheduler

class CodenamesModel(pl.LightningModule):
    def __init__(self, noun_list, vocab_list, word2vec_model, **hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.hparams['vocab_size'] = len(vocab_list)
        self.save_hyperparameters()

        self.transformer_clue = TransformerClue(
            noun_list=noun_list,
            word2vec_model=word2vec_model,
            model_dim=self.hparams['model_dim'],
            vocab_size=len(vocab_list),
            max_guess_count=self.hparams['max_guess_count'],
            num_heads=self.hparams['num_heads'],
            num_layers=self.hparams['num_layers'],
            lr=self.hparams['lr'],
            warmup=self.hparams['warmup_epochs'],
            initial_temperature=self.hparams['initial_temperature'],
            dropout=self.hparams['dropout'],
            input_dropout=self.hparams['input_dropout'],
            min_temperature=self.hparams['min_temperature']
        )

        self.transformer_guesser = TransformerGuesser(
            noun_list=noun_list,
            vocab_list=vocab_list,
            word2vec_model=word2vec_model,
            model_dim=self.hparams['model_dim'],
            max_guess_count=self.hparams['max_guess_count'],
            num_heads=self.hparams['num_heads'],
            num_layers=self.hparams['num_layers'],
            lr=self.hparams['lr'],
            warmup=self.hparams['warmup_epochs'],
            initial_temperature=self.hparams['initial_temperature'],
            dropout=self.hparams['dropout'],
            input_dropout=self.hparams['input_dropout'],
            min_temperature=self.hparams['min_temperature']
        )
        self.temperature = self.hparams['initial_temperature']

    def forward(self, words, classes):
        word_logits = self.transformer_clue(words, classes)
        # clue_weights = gumbel_softmax(word_logits, tau=self.temperature, dim=-1)
        # num_weights = gumbel_softmax(num_logits, tau=self.temperature, dim=-1)
        # clue_weights = F.softmax(word_logits, dim=-1)
        # num_weights = F.softmax(num_logits, dim=-1)
        clue_weights = word_logits
        output = self.transformer_guesser(clue_weights, words)
        if self.trainer.is_last_batch:
            self.log_heatmap(clue_weights, "Class Weights", self.hparams['vocab_list'])
            self.log_true_vs_predicted_heatmap(classes, F.sigmoid(output), "True vs Predicted")
            # self.log_heatmap(num_weights, "Num Weights", list(range(1, self.hparams['max_guess_count'] + 1)))

        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'])

        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams['warmup_epochs'],
                                             max_epochs=self.hparams['max_epochs'],
                                             initial_value=.01,
                                             min_value=.003)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        words, classes = batch
        output = self(words, classes)
        loss = self.compute_loss(output, classes)
        self.log_metrics(output, classes)
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
        correct_count = (correct & classes == 1).sum() / output.shape[0]
        return accuracy_tp.item(), accuracy_tn.item(), correct_count.item()

    def log_metrics(self, predicted, true):
        loss = self.compute_loss(predicted, true).item()
        accuracy_tp, accuracy_tn, correct_count = self.compute_metrics(predicted, true)
        self.log('train_loss', loss)
        self.log('wrong_acc', accuracy_tn)
        self.log('correct_acc', accuracy_tp)
        self.log('correct_count', correct_count)

    def log_heatmap(self, values, name='heatmap', labels=None):
        if values.is_cuda:
            values = values.cpu()

        values = values.detach().numpy()

        # Check if labels are provided, otherwise use default feature numbering
        x_labels = labels if labels is not None else [f"Feature {i+1}" for i in range(values.shape[1])]

        fig = go.Figure(data=go.Heatmap(
            z=values,
            x=x_labels,  # Use custom labels for the x-axis
            y=[f"Batch {i+1}" for i in range(values.shape[0])],  # Naming batches on y-axis
            hoverongaps=False,
            hoverinfo='x+y+z',  # Showing labels and value on hover
            colorscale='Viridis'
        ))

        fig.update_layout(
            title=f"Heatmap for Batch ID {self.current_epoch}",
            xaxis_title="Features",
            yaxis_title="Batch Instances",
            xaxis={'showticklabels': False},  # Hide x-axis labels
            yaxis={'showticklabels': False}   # Hide y-axis labels
        )

        wandb.log({f"{name}_epoch_{self.current_epoch}": wandb.Plotly(fig)})

    def log_true_vs_predicted_heatmap(self, true_values, predicted_values, name='True vs Predicted Heatmap', labels=None):
        if true_values.is_cuda:
            true_values = true_values.cpu()
        if predicted_values.is_cuda:
            predicted_values = predicted_values.cpu()

        true_values = true_values.detach().numpy()
        predicted_values = predicted_values.detach().numpy()

        x_labels = labels if labels is not None else [f"Feature {i+1}" for i in range(true_values.shape[1])]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("True Values", "Predicted Values"))

        fig.add_trace(go.Heatmap(
            z=true_values,
            x=x_labels,
            y=[f"Instance {i+1}" for i in range(true_values.shape[0])],
            colorscale='Viridis',
            zmin=0, zmax=1,
            showscale=False,
            hoverinfo='x+y+z'
        ), row=1, col=1)

        fig.add_trace(go.Heatmap(
            z=predicted_values,
            x=x_labels,
            y=[f"Instance {i+1}" for i in range(predicted_values.shape[0])],
            colorscale='Viridis',
            zmin=0, zmax=1,
            hoverinfo='x+y+z'
        ), row=1, col=2)

        fig.update_layout(
            title=name,
            xaxis_title="Features",
            yaxis_title="Instances",
            xaxis={'showticklabels': False},
            yaxis={'showticklabels': False}
        )

        wandb.log({f"{name}_epoch_{self.current_epoch}": wandb.Plotly(fig)})


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
        self.temperature = cosine_scheduler(self.current_epoch, self.trainer.max_epochs, self.hparams['initial_temperature'], self.hparams['min_temperature'])

    def on_train_epoch_end(self):
        self.update_hp()
        self.log_hp()
