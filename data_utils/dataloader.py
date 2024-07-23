import torch
from torch.utils.data import Dataset, DataLoader


class CodenamesGenerator(Dataset):
    def __init__(self, start_word_index, end_word_index, board_size, max_guesses, device='cpu'):
        self.start_word_index = start_word_index
        self.end_word_index = end_word_index
        self.board_size = board_size
        self.max_guesses = max_guesses
        self.token_size = board_size + 2
        self.device = device

    def __len__(self):
        return self.end_word_index - self.start_word_index

    def __getitem__(self, idx):
        word_tensor = torch.randint(self.start_word_index, self.end_word_index, size=(self.token_size,), device=self.device)
        n = torch.randint(1, self.max_guesses, (1,), device=self.device).item()
        class_tensor = torch.zeros(self.token_size, dtype=torch.int, device=self.device)
        word_tensor[0], word_tensor[1] = 0, 1
        class_tensor[0], class_tensor[1] = 2, 3
        indices = torch.randperm(self.token_size - 2, device=self.device) + 2
        indices = indices[:n]
        class_tensor[indices] = 1
        print("retrieved item")
        return word_tensor, class_tensor

    

def retrieve_data_loader(start_word_index, end_word_index, board_size, max_guesses, batch_size, device='cpu'):
    dataset = CodenamesGenerator(start_word_index, end_word_index, board_size, max_guesses, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
