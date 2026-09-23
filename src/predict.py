import os
import json
import numpy as np
import pandas as pd
import torch

from data_prepare import clean, PARAM_JSON, MODEL, FEATURE_COLS, DATA
from model import TitanicNet

INPUT_CSV = os.path.join(DATA, "input.csv")

def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.drop(columns=["Survived"], errors="ignore")

    with open(PARAM_JSON, "r", encoding="utf-8") as f:
        pm = json.load(f)

    out = clean(df, pm)

    ckpt = torch.load(MODEL, weights_only=False)
    model = TitanicNet(ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = torch.from_numpy(out[FEATURE_COLS].to_numpy(dtype=np.float32))

    with torch.no_grad():
        probability = torch.sigmoid(model(x)).numpy()

    for i in range(len(out)):
        if probability[i] > 0.5:
            print("live")
        else:
            print("death")

if __name__ == "__main__":
    main()