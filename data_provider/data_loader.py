# data_provider/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class RealMTSDataset(Dataset):
    def __init__(self, csv_path, seq_len=96, pred_len=24):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 读取 CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到文件: {csv_path}")
            
        df = pd.read_csv(csv_path)
        if 'date' in df.columns or 'Date' in df.columns:
            df = df.drop(columns=['date', 'Date'], errors='ignore')

        data_matrix = df.values
        self.mean = np.mean(data_matrix, axis=0)
        self.std = np.std(data_matrix, axis=0)
        self.data = (data_matrix - self.mean) / (self.std + 1e-5)
        self.num_features = self.data.shape[1]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    pass

def data_provider(config, flag='train'):
    """统一的数据获取接口"""
    dataset = RealMTSDataset(csv_path=config.csv_path, seq_len=config.seq_len, pred_len=config.pred_len)
    
    # 划分训练/测试
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(config.seed)
    )
    
    if flag == 'train':
        return DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    else:
        return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)