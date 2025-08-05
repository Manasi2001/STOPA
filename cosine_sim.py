"""
USE FOR COMPUTING COSINE SIMILARITY BETWEEN TRIAL AND FINGERPRINT EMBEDDINGS

"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Compute cosine similarity between trial embeddings and fingerprint embeddings ===
fp_avg_emb = pd.read_csv('AASIST_CM_fingerprint_avg_emb.csv')
protocol_trials_data = np.load('AASIST_CM_protocol_trials.npy')  # AA01-co-100_trial.npy abtained from trials_embedding_extraction.py is renamed as AASIST_CM_protocol_trials.npy
protocol_trials = pd.DataFrame(protocol_trials_data)

# Attach file names to the trial embeddings
trial_list = pd.read_csv('datasets/ASVspoof2019_Attribution/protocols_trials/AA01-co-100_trials.txt', sep=" ")
protocol_trials['FileName'] = trial_list['FileName']
# Move 'FileName' column to the front
protocol_trials = protocol_trials[['FileName'] + list(protocol_trials.columns[:-1])]

# Extract embedding matrices
embeddings_50 = fp_avg_emb.iloc[:, 1:].values
embeddings_audio = protocol_trials.iloc[:, 1:].values

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings_audio, embeddings_50)
final_df = pd.DataFrame(similarity_matrix, columns=list(fp_avg_emb['file']))
final_df.insert(0, 'FileName', protocol_trials['FileName'])
final_df.to_csv('AASIST_CM_cosine_similarity_matrix.csv', index=False)

# === 2. Write cosine scores into individual protocol trial files ===
cosine_scores = pd.read_csv('AASIST_CM_cosine_similarity_matrix.csv', index_col=0)
protocols_folder = 'protocols_trials_extended'

for column in cosine_scores.columns:
    file_name = column.replace('-avg', '') + '_trials.txt'
    file_path = os.path.join(protocols_folder, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        with open(file_path, 'w') as f:
            if lines:
                # Add 'CosScore' as the new column header
                f.write(lines[0] + ' CosScore\n')
                for line, score in zip(lines[1:], cosine_scores[column]):
                    f.write(f"{line} {score}\n")
    else:
        print(f'File not found: {file_path}')

print('Cosine scores written to protocol files.')

# === 3. Concatenate selected protocol files into a single CSV ===
output_file = 'AASIST_CM_evaluation.csv'
all_lines = []

for file_name in os.listdir(protocols_folder):
    file_path = os.path.join(protocols_folder, file_name)
    # Filter for relevant text files containing selected attack IDs
    if file_name.endswith('.txt') and os.path.isfile(file_path) and any(p in file_name for p in ['AA01', 'AA03', 'AA05', 'AA07', 'AA10']):
        with open(file_path, 'r') as f:
            lines = [line.strip().replace(' ', ',') for line in f.readlines()]
            all_lines.extend(lines)

# Save the combined data to CSV
with open(output_file, 'w') as f:
    for line in all_lines:
        f.write(line + '\n')

print(f'All protocol files have been concatenated into {output_file}')
