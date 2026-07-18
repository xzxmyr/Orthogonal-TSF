# config.py
class ExperimentConfig:
    # 路径配置
    csv_path = '/Users/xixiangyu/Desktop/03_个人发展与项目/surf/data/ETTm1.csv'
    save_path = 'weight_dynamics.png'
    
    # 架构超参数
    seq_len = 96
    pred_len = 24
    num_features = 7
    hidden_dim = 64
    
    # 训练超参数
    batch_size = 32
    epochs = 8
    lr = 0.001
    lambda_ort = 0.1  # 正交正则化系数
    seed = 42