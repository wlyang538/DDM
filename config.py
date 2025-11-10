# config.py
import os

class Config:
    # ===== path config =====
    base_dir = ""  # main dir
    model_name_or_path = os.path.join(base_dir, "bert-base-uncased")
    
    # datasets dir name (when offline)
    data_root = os.path.join(base_dir, "data")
    sst2_path = os.path.join(data_root, "sst2")
    mr_path = os.path.join(data_root, "mr")
    imdb_path = os.path.join(data_root, "imdb")
    agnews_path = os.path.join(data_root, "ag_news")
    
    # output dir
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ===== parameters for training =====
    max_seq_len = 128
    batch_size = 32
    learning_rate = 5e-5
    num_epochs = 5
    dropout = 0.1
    seed = 42

    # ===== DDM parameters =====
    mask_ratio = 0.15  #  mask ratio when training
    suspicious_ratio = 0.2  # mask ratio when inferencing
