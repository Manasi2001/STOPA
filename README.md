# STOPA: A Dataset of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution

This repository focuses on evaluating open-set source tracing systems for synthetic speech using the STOPA dataset [1]. STOPA (Systematic VariaTion Of DeePfake Audio) is a large, systematically curated dataset with over 700k utterances from 13 synthesis systems, varying across 8 acoustic models and 6 vocoders. It enables precise attribution of deepfake audio to its generative components, supporting real-world forensic and anti-spoofing efforts.

The results presented in Table 4 of the paper [1] benchmark various systems under STOPA’s open-world evaluation protocol, focusing on their ability to attribute audio to the correct **attack type (ATK)**, **acoustic model (AM)**, or **vocoder model (VM)**. Three systems are compared: a ResNet-34 model [3], an ASVspoof2019 [4]-trained countermeasure (AASIST CM), and a STOPA-trained AASIST [2] model. The error rates (EER%) highlight the inherent difficulty of the task, especially under unknown attack conditions, where attribution performance approaches chance levels. This reinforces the core motivation behind STOPA: promoting the design of scalable and adaptive profiling systems that do not require retraining when new deepfake generation methods emerge.

### Directory Structure

```
.
├── AASIST_CM_EERs_pooled/               # Pooled EER CSVs for AASIST_CM
├── AASIST_CM_embeddings/                # Fingerprint embeddings for each spoof condition (AASIST_CM)
├── AASIST_CM_evaluation.csv             # Score labels and cosine similarity (AASIST_CM)
├── AASIST_CM_fingerprint_avg_emb.csv    # Averaged fingerprint embeddings (AASIST_CM)
├── AASIST_CM_protocol_trials.npy        # Trial protocol data for AASIST_CM
│
├── AASIST_STOPA_EERs_pooled/            # Pooled EER CSVs for AASIST_STOPA
├── AASIST_STOPA_embeddings/             # Fingerprint embeddings for each spoof condition (AASIST_STOPA)
├── AASIST_STOPA_evaluation.csv          # Score labels and cosine similarity (AASIST_STOPA)
├── AASIST_STOPA_fingerprint_avg_emb.csv # Averaged fingerprint embeddings (AASIST_STOPA)
├── AASIST_STOPA_protocol_trials.npy     # Trial protocol data for AASIST_STOPA
│
├── ResNet-34_STOPA_EERs_pooled/         # Pooled EER CSVs for ResNet-34
├── ResNet-34_STOPA_evaluation.csv       # Evaluation results for ResNet-34
│
├── models/                              # Trained model checkpoints
│   ├── AASIST_CM.pth
│   ├── AASIST_STOPA.pth
│   └── ResNet-34.pth
│
├── config/                              # Model config files
│   ├── AASIST.conf
│   └── ResNet-34.conf
│
├── protocols_fingerprint_extraction/    # Protocol files for fingerprint creation
├── protocols_trials_extended/           # Protocol files for trial embedding extraction
│
├── compute_eer_pooled.py                # Script to compute pooled EERs
├── cosine_sim.py                        # Compute cosine similarities between fingerprint and trial embeddings
├── create_fingerprint_avg_emb_csv.py    # Averaging fingerprint embeddings
├── fingerprint_embedding_extraction.py  # Extract fingerprint embeddings
├── trials_embedding_extraction.py       # Extract embeddings for trial utterances

```
### Dataset Preparation

The STOPA dataset is available at https://zenodo.org/records/15606628. Please follow the instructions on the page to download and extract the corpus. In this repository, the `TEE` folder from STOPA corresponds to `protocols_fingerprint_extraction/`, and the `Trials` folder maps to `protocols_trials_extended/`. Place these folders accordingly before running the embedding and evaluation scripts.

### Step 1: Extract Fingerprint Embeddings

```python fingerprint_embedding_extraction.py```

- Reads protocol files from `protocols_fingerprint_extraction/`.

- Outputs embeddings to `AASIST_CM_embeddings/`, `AASIST_STOPA_embeddings/`, etc.

### Step 2: Create Averaged Fingerprint Embeddings

```python create_fingerprint_avg_emb_csv.py```

- Computes mean embeddings per spoof condition.

- Outputs to `AASIST_CM_fingerprint_avg_emb.csv`, `AASIST_STOPA_fingerprint_avg_emb.csv`.

### Step 3: Extract Trial Embeddings

```python trials_embedding_extraction.py```

- Uses `protocols_trials_extended/` to extract trial utterances.

- Outputs trial embeddings and trial protocol files (`*_protocol_trials.npy`).

### Step 4: Compute Cosine Similarity

```python cosine_sim.py```

- Computes cosine scores between trial and fingerprint embeddings.

- Outputs evaluation results as `*_evaluation.csv`.

### Step 5: Compute Pooled EER

```python compute_eer_pooled.py```

- Uses evaluation CSVs and protocol files to compute pooled EER.

- Outputs `pooled_eer_kn.csv` and `pooled_eer_ukn.csv` under each system folder.

### Output

After running the above steps, you'll get the following files for all spoofing conditions.  

```AASIST_CM_EERs_pooled/pooled_eer_kn.csv```

```AASIST_CM_EERs_pooled/pooled_eer_ukn.csv```

```AASIST_STOPA_EERs_pooled/pooled_eer_kn.csv```

```AASIST_STOPA_EERs_pooled/pooled_eer_ukn.csv```

```ResNet-34_STOPA_EERs_pooled/pooled_eer_kn.csv```

```ResNet-34_STOPA_EERs_pooled/pooled_eer_ukn.csv```

### License

```
MIT License

Copyright (c) 2025 Manasi Chhibber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Acknowledgements

- This work was partially supported by the Brno University of Technology (internal project FIT-S-23-8151) and the Academy of Finland (DecisionNo. 349605, project ”SPEECHFAKES”). The authors wish to acknowledge CSC – IT Center for Science, Finland, for computational resources.
- Script for training the AASIST model can be found at https://github.com/clovaai/aasist.

### References

[1] STOPA: A Database of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution
```bibtex
@misc{firc2025stopadatabasesystematicvariation,
author={Anton Firc and Manasi Chhibber and Jagabandhu Mishra and Vishwanath Pratap Singh and Tomi Kinnunen and Kamil Malinka},
year={2025},
eprint={2505.19644},
archivePrefix={arXiv},
primaryClass={cs.SD},
url={https://arxiv.org/abs/2505.19644}, 
}
```

[2] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}
```

[3] ASVspoof 5 Challenge: advanced ResNet architectures for robust voice spoofing detection
```bibtex
@inproceedings{dao2024asvspoof,
  author={Dao, Anh-Tuan and Rouvier, Mickael and Matrouf, Driss},
  booktitle={Proc. The Automatic Speaker Verification Spoofing Countermeasures Workshop (ASVspoof Workshop)},
  pages={163--169},
  year={2024}
}
```

[4] ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```