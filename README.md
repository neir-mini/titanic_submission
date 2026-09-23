# 泰坦尼克号生存预测（PyTorch）

用 PyTorch 自建的神经网络预测泰坦尼克号乘客是否幸存。

- **任务**：二分类（幸存 / 遇难）
- **数据**：891 行 × 12 列，整体幸存率 38.38%
- **实现**：`torch.nn.Module` 自建网络，未使用任何现成的分类器
- **结果**：测试集准确率 **80.45%**（瞎猜基线 58.65%，泛化差距 2.91%）

## 目录结构

```
titanic_submission/
├── data/
│   ├── raw_data.csv        891 行原始数据（唯一的数据来源）
│   ├── train.csv           训练集 625 行
│   ├── val.csv             验证集 133 行
│   ├── test.csv            测试集 133 行
│   ├── input.csv           要预测的数据（需自行放入）
│   └── prep_params.json    预处理参数（训练/预测共用，保证一致）
├── src/
│   ├── data_prepare.py     清洗 + 特征工程 + 切分三份
│   ├── model.py            网络结构（18 行）
│   ├── train.py            训练循环 + 画图 + 保存模型
│   ├── predict.py          加载模型，对 input.csv 输出 live / death
│   └── requirements.txt    依赖清单
├── docs/
│   ├── 学习笔记.pdf         实现思路与原理笔记
│   ├── loss_curve.png      损失曲线
│   └── accuracy_curve.png  准确率曲线（train 与 val 同图）
└── outputs/
    └── model.pth           训练好的模型（含 state_dict 与 n_features）
```

## 运行方法

```bash
pip install -r src/requirements.txt

cd src
python data_prepare.py    # 1. 生成 train/val/test.csv + prep_params.json
python train.py           # 2. 训练，生成 outputs/model.pth + docs/ 两张图
python predict.py         # 3. 需先在 data/ 放好 input.csv
```

`predict.py` 会在终端逐行打印 `live` 或 `death`。`input.csv` 至少需要包含
`Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked, Name` 这些列
（`Cabin` 可以为空但列必须存在，`Survived` 可有可无）。

## 特征工程

原始 12 列中，`PassengerId` / `Ticket` 丢弃，`Cabin` 缺失 77% 所以只保留
"是否有客舱号"这一位，其余处理如下，最终得到 12 个特征：

| 特征 | 处理方式 |
|---|---|
| `Sex` | male→0，female→1 |
| `Age` | 缺失用训练集中位数 28.0 填补 |
| `Fare` | 缺失用中位数填补后，用训练集均值方差标准化 |
| `Embarked` | 缺失填众数 S，再 S→0 / C→1 / Q→2 |
| `HasCabin` | `Cabin` 非空为 1 |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `FamilySize == 1` |
| `Title` | 从 `Name` 正则提取称呼，17 种归并为 6 类 |
| `IsChild` | `Age <= 12` |

所有类别特征都使用固定的 `map` 字典而非独热编码，保证特征列的数量和顺序
在任何输入下都不变。

## 模型与训练

```
Linear(12, 16) → ReLU → Dropout(0.2) → Linear(16, 1)
```

- 参数量：**225**（远小于 625 个训练样本）
- 损失：`BCEWithLogitsLoss`
- 优化器：`Adam(lr=1e-3, weight_decay=1e-4)` + 余弦退火
- 训练：300 轮，批大小 32
- 随机种子固定为 42，结果可复现
- 训练结束后载入"验证集准确率最高"那一轮的参数

## 结果

| 指标 | 数值 |
|---|---|
| 训练集准确率 | 83.36% |
| 验证集准确率 | 84.21% |
| **测试集准确率** | **80.45%** |
| 瞎猜基线 | 58.65% |
| 泛化差距 | 2.91% |

## 一点说明

测试集只有 133 个样本，**每错 1 个样本准确率就变动 0.75 个百分点**。
换 6 个不同的随机种子，测试集准确率在 76.69% ~ 85.71% 之间波动
（均值 80.45%）。因此 80.45% 应理解为"约 80% 上下 4 个点"，
而不是一个精确值。这也是本项目没有继续加深网络的原因——把网络从
2 层加到 4 层（参数量 225 → 961）在 5 个种子上的平均提升只有约
0.3 个百分点，属于噪声范围，瓶颈在数据规模而非模型容量。
