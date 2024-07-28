import torch
from torch.utils.data import Dataset, DataLoader


class CodenamesGenerator(Dataset):
    def __init__(self, start_word_index, end_word_index, board_size, max_guesses, device='cpu'):
        self.start_word_index = start_word_index
        self.end_word_index = end_word_index
        self.board_size = board_size
        self.max_guesses = max_guesses
        self.token_size = board_size
        self.device = device

    def __len__(self):
        return 80000

    def __getitem__(self, idx):
        word_tensor = torch.randint(self.start_word_index, self.end_word_index, size=(self.token_size,), device=self.device)
        n = torch.randint(1, self.max_guesses+1, (1,), device=self.device).item()
        class_tensor = torch.zeros(self.token_size, dtype=torch.int, device=self.device)
        indices = torch.randperm(self.token_size, device=self.device)
        indices = indices[:n]
        class_tensor[indices] = 1
        return word_tensor, class_tensor

    

def retrieve_data_loader(start_word_index, end_word_index, board_size, max_guesses, batch_size, num_workers=0, device='cpu'):
    dataset = CodenamesGenerator(start_word_index, end_word_index, board_size, max_guesses, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
