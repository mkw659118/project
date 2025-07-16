import pandas as pd
import torch
import os
import pickle

# ========== 参数配置 ==========
seq_len = 32
pred_len = 1

target_column = "Close"
model_path = f"./model/model_{target_column}_1.bin"

output_top10 = "./output/result1.csv"
output_all_preds = "./output/all_predictions1.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 数据加载与预处理 ==========
def inputdata(path):
    return pd.read_csv(path, header=0, sep=",", encoding="utf-8")


def transcolname(df, column_mapping):
    df.rename(columns=column_mapping, inplace=True)
    return df


def trans_datetime(df):
    dt = df["Date"]
    df_time = pd.DataFrame()
    df_time["year"] = dt.transform(lambda x: int(x.split("-")[0]))
    df_time["month"] = dt.transform(lambda x: int(x.split("-")[1]))
    df_time["day"] = dt.transform(lambda x: int(x.split("-")[2][:2]))
    df = pd.concat([df, df_time], axis=1)
    unique_dates = pd.Series(df["Date"].unique()).sort_values().reset_index(drop=True)
    date_mapping = {d: i + 1 for i, d in enumerate(unique_dates)}
    df["Date"] = df["Date"].map(date_mapping)
    return df


def processing_feature_test():
    data = inputdata("./data/test.csv")
    column_mapping = {
        "股票代码": "StockCode",
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Turnover",
        "振幅": "Amplitude",
        "涨跌额": "PriceChange",
        "换手率": "TurnoverRate",
        "涨跌幅": "PriceChangePercentage",
    }
    data = transcolname(data, column_mapping)

    if "PriceChangePercentage" in data.columns:
        data.drop(columns=["PriceChangePercentage"], inplace=True)
    data = trans_datetime(data)
    return data


# ========== 加载数据 ==========
data_raw = processing_feature_test()

column_names = data_raw.columns.tolist()
colname2index = {x: i for i, x in enumerate(column_names)}
stockcodes = data_raw["StockCode"].drop_duplicates().tolist()
target_index = colname2index["Close"]
print(colname2index["Close"])
max_date = data_raw["Date"].max()

# ========== 加载模型 ==========
assert os.path.exists(model_path), f"模型文件不存在: {model_path}"
model = pickle.load(open(model_path, "rb")).to(device)
model.eval()

# ========== 预测 + 排序 + 保存 ==========
all_preds = []
all_records = []

for stockcode in stockcodes:
    raw_data = data_raw[data_raw["StockCode"] == stockcode].sort_values("Date")
    if len(raw_data) < seq_len:
        continue

    raw_input = raw_data.iloc[-seq_len:]

    try:
        with open('./x_scaler1.pkl', 'rb') as f:
            x_scaler = pickle.load(f)

        with open('./y_scaler1.pkl', 'rb') as f:
            y_scaler = pickle.load(f)

    except Exception as e:
        print(e)

    input_tensor = torch.as_tensor(x_scaler.transform(raw_input), dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        pred = model(input_tensor, None, None, None).cpu().numpy().squeeze(0)
        pred = y_scaler.inverse_transform(pred)

    true_close = raw_input.iloc[-1]["Close"]
    date = raw_input.iloc[-1]["Date"]

    # pred 是 (1, feature_dim) 或 (feature_dim,) 形状的数组
    # 提取 Close 这一列对应的预测值
    pred_value = pred[0] if pred.ndim == 1 else pred[0, target_index]

    # 然后计算涨跌幅
    change_rate = (pred_value - true_close) / true_close * 100

    record = {
        "StockCode": stockcode,
        "Date": date,
        "True_Close": true_close,
        "Pred_Close": pred_value,
        "Change_%": change_rate
    }

    all_records.append(record)
    all_preds.append((stockcode, change_rate))

# ========== 保存 Top10 涨跌幅 ==========
all_preds = sorted(all_preds, key=lambda x: x[1], reverse=True)
pred_top_10_max_target = [x[0] for x in all_preds[:10]]
pred_top_10_min_target = [x[0] for x in all_preds[-10:]]
print(all_preds[:10])
print(all_preds[-10:])
os.makedirs(os.path.dirname(output_top10), exist_ok=True)
result_df = pd.DataFrame({
    "涨幅最大股票代码": pred_top_10_max_target,
    "涨幅最小股票代码": pred_top_10_min_target,
})
result_df.to_csv(output_top10, index=False)
print(f"✅ 涨跌幅Top10结果已保存：{output_top10}")

# ========== 保存所有预测记录 ==========
df_all = pd.DataFrame(all_records)
df_all.to_csv(output_all_preds, index=False)
print(f"✅ 所有股票预测记录已保存：{output_all_preds}")
