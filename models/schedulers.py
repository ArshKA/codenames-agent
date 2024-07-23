import math
from torch import optim

def cosine_scheduler(epoch, num_epochs, initial_value, min_value=0.1):
    return min_value + (initial_value - min_value) * 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_epochs, initial_value, min_value):
        self.warmup = warmup
        self.max_epochs = max_epochs
        self.initial_value = initial_value
        self.min_value = min_value
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = cosine_scheduler(epoch, self.max_epochs, self.initial_value, self.min_value)
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor