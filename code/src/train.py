import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from argparse import Namespace
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import numpy as np
from DLinear import Model as DLinearModel
from transformer import Transformer


# ========== EarlyStopping 类 ==========
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = 0
        self.best_model = None
        self.best_epoch = None

    def __call__(self, epoch, params, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch, params, val_loss)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_epoch = epoch
            self.best_score = score
            self.save_checkpoint(epoch, params, val_loss)
            self.counter = 0

    def save_checkpoint(self, epoch, params, val_loss):
        self.best_epoch = epoch + 1
        self.best_model = params
        self.val_loss_min = val_loss

    def track(self, epoch, params, error):
        self.__call__(epoch, params, error)

    def track_one_epoch(self, epoch, model, error):
        self.track(epoch, model.state_dict(), error)


# ========== 设备选择 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")


# ========== 数据加载 ==========
def inputdata(path):
    return pd.read_csv(path, header=0, sep=",", encoding="utf-8")


feature_raw = inputdata("./temp/feature.csv")
column_names = feature_raw.columns.tolist()
colname2index = {x: i for i, x in enumerate(column_names)}
enc_in = feature_raw.shape[1]
target_column = "Close"
target_index = colname2index[target_column]
stockcodes = feature_raw["StockCode"].drop_duplicates().tolist()


# ========== 构造样本函数 ==========
def process_data(npdf, stp=32, pred_len=1):
    ret = []
    for i in range(npdf.shape[0] - stp - pred_len + 1):
        seq = npdf[i: i + stp, :]
        label = npdf[i + stp: i + stp + pred_len, :]
        seq = torch.FloatTensor(seq)
        label = torch.FloatTensor(label)
        ret.append((seq, label))
    return ret


# ========== 划分训练集和验证集 ==========
train_data, val_data = [], []

for stockcode in stockcodes:
    stock_df = feature_raw[feature_raw["StockCode"] == stockcode]
    if len(stock_df) < 33:
        continue
    split_idx = int(len(stock_df) * 0.8)
    train_part = stock_df.iloc[:split_idx].values
    val_part = stock_df.iloc[split_idx:].values

    train_data += process_data(train_part, stp=32, pred_len=1)
    val_data += process_data(val_part, stp=32, pred_len=1)


def generate_causal_mask(seq_len, pred_len, device):
    total_len = seq_len + pred_len
    mask = torch.triu(torch.ones(total_len, total_len), diagonal=1).bool()
    return mask.to(device)


# ========== 模型训练函数 ==========
def train_model(train_data, val_data, win_size, enc_in, num_epochs=100):
    if len(train_data) == 0 or len(val_data) == 0:
        return None

    train_data = [(x.to(device), y.to(device)) for x, y in train_data]
    val_data = [(x.to(device), y.to(device)) for x, y in val_data]

    X_train = torch.stack([x for x, _ in train_data])
    Y_train = torch.stack([y for _, y in train_data])

    X_cpu = X_train.cpu().reshape(-1, X_train.shape[-1])
    Y_cpu = Y_train.cpu().reshape(-1, Y_train.shape[-1])

    # x_scaler = StandardScaler().fit(X_cpu)
    # y_scaler = StandardScaler().fit(Y_cpu)

    x_scaler = MinMaxScaler().fit(X_cpu)
    y_scaler = MinMaxScaler().fit(Y_cpu)

    X_val = torch.stack([x for x, _ in val_data])
    Y_val = torch.stack([y for _, y in val_data])

    X_train = torch.tensor(x_scaler.transform(X_cpu).reshape(X_train.shape), dtype=torch.float32).to(device)
    Y_train = torch.tensor(y_scaler.transform(Y_cpu).reshape(Y_train.shape), dtype=torch.float32).to(device)

    X_val = torch.tensor(x_scaler.transform(X_val.cpu().reshape(-1, X_val.shape[-1])).reshape(X_val.shape), dtype=torch.float32).to(device)
    Y_val = torch.tensor(y_scaler.transform(Y_val.cpu().reshape(-1, Y_val.shape[-1])).reshape(Y_val.shape), dtype=torch.float32).to(device)

    with open('./x_scaler1.pkl', 'wb') as f:
        pickle.dump(x_scaler, f)
    with open('./y_scaler1.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)
    print('✅ StandardScaler 已保存')

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=32, shuffle=False)

    # config = Namespace(task_name='long_term_forecast', seq_len=win_size, pred_len=1, enc_in=enc_in, moving_avg=25)
    # model = DLinearModel(configs).to(device)
    config = Namespace(seq_len=win_size, num_layers=2, n_heads=4, pred_len=1, enc_in=enc_in, revin=True, d_model=64)
    model = Transformer(
        input_size=config.enc_in,
        d_model=config.d_model,
        revin=config.revin,
        num_heads=config.n_heads,
        num_layers=config.num_layers,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    monitor = EarlyStopping(patience=15)

    print("\n🚀 Start training DLinear model with validation support")
    for epoch in range(num_epochs):
        if monitor.early_stop:
            break
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", unit="batch")
        for batch_X, batch_y in pbar:
            optimizer.zero_grad()
            mask = generate_causal_mask(seq_len=win_size, pred_len=1, device=batch_X.device)
            output = model(batch_X, attention_mask=mask)

            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        for val_X, val_y in val_loader:
            mask = generate_causal_mask(seq_len=win_size, pred_len=1, device=val_X.device)
            val_output = model(val_X, attention_mask=mask)
            loss = criterion(val_output, val_y)
            val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        monitor.track(epoch, model.state_dict(), avg_val_loss)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    print(f"\n✅ Training complete. Best model at epoch {monitor.best_epoch} with Val Loss: {monitor.val_loss_min:.6f}")
    model.load_state_dict(monitor.best_model)
    return model


# ========== 启动训练并保存 ==========
model_i = train_model(train_data, val_data, win_size=32, enc_in=enc_in, num_epochs=100)
model_name = f"./model/model_{target_column}_1.bin"
pickle.dump(model_i, open(model_name, "wb"))
print(f"✅ Model saved to {model_name}")
