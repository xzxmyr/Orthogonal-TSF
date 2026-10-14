from pathlib import Path

# config.py
class ExperimentConfig:
    csv_path = str(Path(__file__).resolve().parents[1] / 'data' / 'ETTm1.csv')
    save_path = 'weight_dynamics.png'
    
    seq_len = 96
    pred_len = 24
    num_features = 7
    hidden_dim = 64
    
    batch_size = 32
    epochs = 8
    lr = 0.001
    
    # ==== 全新条件信息论超参数 ====
    lambda_cmi = 0.001       # CMI最小化惩罚权重 (替换原 lambda_ort)
    mine_lr = 0.0005        # MINE 判别器独立学习率
    seed = 42
