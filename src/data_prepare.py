import os
import json
import numpy as np
import pandas as pd

# 1.路径与配置
# 路径
SRC = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SRC)
DATA = os.path.join(PROJECT,"data")
RAW_DATA_CSV = os.path.join(DATA,"raw_data.csv")
PARAM_JSON = os.path.join(DATA, "prep_params.json")
DOCS = os.path.join(PROJECT, "docs")
OUTPUT = os.path.join(PROJECT, "outputs")
MODEL = os.path.join(OUTPUT, "model.pth")

# 数据切分
SEED = 42
VAL_RATIO = 0.15       # 15% 做验证集
TEST_RATIO = 0.15      # 15% 做测试集

# 超参数
N_FEATURES = 12
LABEL_COL = "Survived"
EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
HIDDEN = 16

# 2.字典和映射表
TITLE_MAP = {
    "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
    "Dr": 4, "Rev": 4, "Col": 5, "Major": 5, "Capt": 5, "Mlle": 1,
    "Ms": 1, "Mme": 2, "Countess": 5, "Lady": 5, "Sir": 5, "Jonkheer": 5,
    "Don": 5, "Dona": 5,
}
# 原始数据里有 17 种称呼，但很多只出现 1~2 次（Jonkheer、Countess 各只 1 人）
#   样本太少，模型学不到规律，反而添乱
#       0 Mr(成年男性)    517人  存活率 15.7%     3 Master(小男孩) 40人  存活率 57.5%
#       1 Miss/Ms       185人  存活率 70.3%     4 Dr/Rev（高级职业者）         13人  存活率 23.1%
#       2 Mrs           126人  存活率 79.4%     5 贵族/军官      10人  存活率 50.0%
FEATURE_COLS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "HasCabin", "FamilySize", "IsAlone", "Title", "IsChild",
]

# 3.清洗函数
def clean(df, pm):
  out = pd.DataFrame()
  out["Pclass"] = df["Pclass"]        
  out["SibSp"] = df["SibSp"]          
  out["Parch"] = df["Parch"]
  out["Sex"] = df["Sex"].map({"male": 0, "female": 1})
  out["Age"] = df["Age"].fillna(pm["age_median"])
  out["Fare"] = df["Fare"].fillna(pm["fare_median"])
  out["Embarked"] = df["Embarked"].fillna(pm["embarked_mode"]).map({"S": 0, "C": 1, "Q": 2})
  out["HasCabin"] = df["Cabin"].notnull().astype(int)
  out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
  out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
  out["Title"] = (
    df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    .map(TITLE_MAP).fillna(0).astype(int)
  )
  out["IsChild"] = (out["Age"] <= 12).astype(int)
  out["Fare"] = (out["Fare"] - pm["fare_mean"]) / (pm["fare_std"] + 1e-8)
  if LABEL_COL not in df.columns:
    return out[FEATURE_COLS]
  out["Survived"] = df["Survived"]
  out = out[FEATURE_COLS + ["Survived"]]
  return out

# 4.主流程
# 4.1算预处理参数，用于补全原始数据
def main():
    np.random.seed(SEED)
    df = pd.read_csv(RAW_DATA_CSV)
    print(f"[prepare] 读入 {RAW_DATA_CSV}")

    pm = {
          "age_median": float(df["Age"].median()),
          "fare_median": float(df["Fare"].median()),
          "fare_mean": float(df["Fare"].mean()),
          "fare_std": float(df["Fare"].std()),
          "embarked_mode": str(df["Embarked"].mode()[0])
    }
    
    # 4.2清洗
    out = clean(df, pm)
    print("数据清理中")
    assert not out.isnull().any().any(), "还有缺失值！"
    print("数据清理完毕！\n启动！！！")
    # 4.3切分成三份
    n = len(out)
    idx = np.random.permutation(n)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    splits = {
        "train": out.iloc[train_idx].reset_index(drop=True),
        "val": out.iloc[val_idx].reset_index(drop=True),
        "test": out.iloc[test_idx].reset_index(drop=True),
    }
    # 4.4保存
    for name, part in splits.items():
        p = os.path.join(DATA, f"{name}.csv")
        part.to_csv(p, index=False)
        print(f"[prepare] {name:<5} 已保存到 {p}")
    with open(PARAM_JSON, "w", encoding="utf-8") as f:
      json.dump(pm, f, ensure_ascii=False, indent=2)
      print(f"[prepare] 预处理参数已保存到 {PARAM_JSON}")

if __name__ == "__main__":
    main()      

