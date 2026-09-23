import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prepare import (SEED, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY,
                          DROPOUT, HIDDEN, N_FEATURES, LABEL_COL,
                          DATA, DOCS, OUTPUT, MODEL)
from model import TitanicNet                          

# 随机种子（random，np，torch种子算法不一样）
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# 读入csv文件，并将其转化为tensor
def read_csv(name):
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"))
    X = df.iloc[:, :N_FEATURES].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.float32)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

# 计算损失函数与准确率
@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    logits = model(X)
    loss = model.criterion(logits, y).item()
    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == y).float().mean().item()
    return loss, acc

# 训练
def train():
    set_seed(SEED)

    Xtr, ytr = read_csv("train")
    Xva, yva = read_csv("val")
    Xte, yte = read_csv("test")
    model = TitanicNet(N_FEATURES)
    total_params = sum(p.numel() for p in model.parameters())
    
    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    best_acc, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = model.criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        train_loss = total_loss / len(Xtr)
        _, train_acc = evaluate(model, Xtr, ytr)
        val_loss, val_acc = evaluate(model, Xva, yva)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)

    return model, hist, (Xtr, ytr), (Xva, yva), (Xte, yte), total_params

# 绘图
def plot(hist):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ep = range(1, EPOCHS + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(ep, hist["train_loss"], label="Train Loss")
    plt.plot(ep, hist["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(DOCS, "loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ep, hist["train_acc"], label="Train Accuracy")
    plt.plot(ep, hist["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(DOCS, "accuracy_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

# 保存模型
def save_model(model):
    torch.save({"state_dict": model.state_dict(), "n_features": N_FEATURES}, MODEL)

# 主流程
def main():
    model, hist, tr, va, te, total_params = train()

    tr_loss, tr_acc = evaluate(model, *tr)
    va_loss, va_acc = evaluate(model, *va)
    te_loss, te_acc = evaluate(model, *te)
    base = max(te[1].mean().item(), 1 - te[1].mean().item())

    plot(hist)
    save_model(model)

    print(f"参数量        {total_params}")
    print(f"训练集准确率  {tr_acc*100:.2f}%")
    print(f"验证集准确率  {va_acc*100:.2f}%")
    print(f"测试集准确率  {te_acc*100:.2f}%")
    print(f"瞎猜基线      {base*100:.2f}%")
    print(f"泛化差距      {(tr_acc-te_acc)*100:.2f}%")

if __name__ == "__main__":
    main()