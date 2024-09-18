import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from data_process.CustomDataset import CustomDataset
from model.patchtst.patchTST import PatchTST
from data_process.DataModule import DataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_scaler, output_scaler, train_dataloader, test_dataloader = DataModule()

