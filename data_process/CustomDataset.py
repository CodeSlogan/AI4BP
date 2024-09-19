import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SequenceDataset(Dataset):
    def __init__(self, input_seq1, input_seq2, output_seq):
        """
        input_seq1: NumPy array, shape (N, 1024)
        input_seq2: NumPy array, shape (N, 1024)
        output_seq: NumPy array, shape (N, 1024)
        """
        self.input_seq1 = input_seq1
        self.input_seq2 = input_seq2
        self.output_seq = output_seq

    def __len__(self):
        # 返回数据集的大小
        return len(self.input_seq1)

    def __getitem__(self, idx):
        # 返回单个样本
        input1 = torch.tensor(self.input_seq1[idx], dtype=torch.float32)
        input2 = torch.tensor(self.input_seq2[idx], dtype=torch.float32)
        output = torch.tensor(self.output_seq[idx], dtype=torch.float32)

        input_combine = torch.stack((input1, input2), dim=0)

        return input_combine, output


class SequenceDatasetPPG(Dataset):
    def __init__(self, input_seq1, input_seq2, output_seq):
        """
        input_seq1: NumPy array, shape (N, 1024)
        input_seq2: NumPy array, shape (N, 1024)
        output_seq: NumPy array, shape (N, 1024)
        """
        self.input_seq1 = input_seq1
        self.input_seq2 = input_seq2
        self.output_seq = output_seq

    def __len__(self):
        # 返回数据集的大小
        return len(self.input_seq1)

    def __getitem__(self, idx):
        # 返回单个样本
        input1 = torch.tensor(self.input_seq1[idx], dtype=torch.float32)
        # input2 = torch.tensor(self.input_seq2[idx], dtype=torch.float32)
        output = torch.tensor(self.output_seq[idx], dtype=torch.float32)

        # input_combine = torch.stack((input1, input2), dim=0)

        return input1, output
