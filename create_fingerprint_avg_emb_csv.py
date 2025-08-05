"""
USE FOR COMPUTING AVG OF FINGERPRINT EMBEDDINGS

"""

import pandas as pd
import numpy as np
import os

embeddings_dir = "AASIST_CM_embeddings"   # modify embeddings directory path as required

target_files = {"AA01", "AA03", "AA05", "AA07", "AA10"}

df_final = pd.DataFrame()

for filename in os.listdir(embeddings_dir):
    for prefix in target_files:
        if filename.startswith(prefix) and filename.endswith(".npy"):
            file_path = os.path.join(embeddings_dir, filename)
            data = np.load(file_path)

            df = pd.DataFrame(data)
            col_avg = df.mean(axis=0)

            df_avg = pd.DataFrame(col_avg).T
            df_avg.insert(0, 'file', filename.replace('.npy', ''))

            df_final = pd.concat([df_final, df_avg], ignore_index=True)

df_final.to_csv("AASIST_CM_fingerprint_avg_emb.csv", index=False)
print("Averages saved!")