'''
USE FOR COMPUTING EER.

'''

import warnings
import torch
import pandas as pd
from pathlib import Path
from evaluation_v1 import compute_eer

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("AASIST_CM_evaluation.csv")

# Known attack types
kn_attk = ["AA01", "AA03", "AA05", "AA07", "AA10"]

# Pooling utterances across all durations
pooled_data = df.copy()

def compute_pooled_eer(df_temp, label):
    true_scores = df_temp[df_temp[label] == True]["CosScore"].to_numpy()
    false_scores = df_temp[df_temp[label] == False]["CosScore"].to_numpy()
    eer, _ = compute_eer(true_scores, false_scores)
    return eer

# Compute pooled EER for known and unknown attack scenarios
eer_kn = {}
eer_ukn = {}

# Known attack scenario
temp_kn = pooled_data[pooled_data["ATK"].isin(kn_attk)].copy()
eer_kn["ATK"] = compute_pooled_eer(temp_kn, "IsTargetATK")
eer_kn["AcousticModel"] = compute_pooled_eer(temp_kn, "IsTargetAcousticModel")
eer_kn["VocoderModel"] = compute_pooled_eer(temp_kn, "IsTargetVocoderModel")

del temp_kn

# Unknown attack scenario
temp_ukn = pooled_data[~pooled_data["ATK"].isin(kn_attk) | (pooled_data["IsTargetATK"] == True)].copy()
eer_ukn["ATK"] = compute_pooled_eer(temp_ukn, "IsTargetATK")
eer_ukn["AcousticModel"] = compute_pooled_eer(temp_ukn, "IsTargetAcousticModel")
eer_ukn["VocoderModel"] = compute_pooled_eer(temp_ukn, "IsTargetVocoderModel")

del temp_ukn

# Save results
exp_dir = Path("AASIST_CM_EERs_pooled")
exp_dir.mkdir(parents=True, exist_ok=True)

def save_eer(eer_dict, filename):
    df = pd.DataFrame.from_dict(eer_dict, orient="index", columns=["EER"])
    df["EER"] = (df["EER"] * 100).round(2)  # Multiply by 100 and round to 2 decimals
    df.to_csv(exp_dir / filename)

save_eer(eer_kn, "pooled_eer_kn.csv")
save_eer(eer_ukn, "pooled_eer_ukn.csv")