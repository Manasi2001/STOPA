"""
USE FOR EXTRACTING TRIALS EMBEDDINGS

"""

import json
import os
import warnings
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from importlib import import_module
import numpy as np
import torch.nn.functional as F

from data_utils import Dataset_Custom

warnings.filterwarnings("ignore", category=FutureWarning)

def collate_fn(batch):
    batch_x, utt_ids = zip(*batch)
    max_len = max(x.shape[1] for x in batch_x)
    batch_x_padded = [F.pad(x.squeeze(0), (0, max_len - x.shape[1])) for x in batch_x]
    batch_x_padded = torch.stack(batch_x_padded).unsqueeze(1)
    return batch_x_padded, utt_ids

def get_custom_loader(protocol_dir: Path, config: dict, wav_dir: Path):
    protocol_files = sorted([file for file in protocol_dir.glob("AA01-co-100_trial*.txt") if file.is_file()])  
    loaders = {}
    for file in protocol_files:
        data = []
        with open(file, 'r') as f:
            lines = f.readlines()[1:]  
            for line in lines:
                parts = line.strip().split()
                attack_type = parts[4]
                filename = parts[1]
                data.append((str(Path(attack_type) / "eval" / filename)))
                
        dataset = Dataset_Custom(data, base_dir=wav_dir)
        loaders[file.stem] = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                        drop_last=False, pin_memory=True, collate_fn=collate_fn)
    return loaders

def get_model(model_config: dict, device: torch.device):
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    print("Model loaded with {} parameters".format(sum(p.numel() for p in model.parameters())))
    return model

def generate_embeddings(data_loader: DataLoader, model, device: torch.device):
    model.eval()
    embs = torch.tensor([], device=device)
    for batch_x, _ in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            emb, _ = model(batch_x.squeeze(1))
            embs = torch.cat((embs, emb), dim=0)
    return embs.cpu().numpy()

def save_embeddings(protocol_dir: Path, wav_dir: Path, model, device, output_dir: Path, config: dict):
    loaders = get_custom_loader(protocol_dir, config, wav_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_name, loader in loaders.items():
        embeddings = generate_embeddings(loader, model, device)
        np.save(output_dir / f"{file_name}.npy", embeddings)
        print(f"Saved embeddings: {file_name}.npy")

if __name__ == "__main__":
    config_path = 'config/AASIST.conf'     # modify configuration file path here for AASIST/ResNet-34 model
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())
    
    emb_config = config["embd_config"]
    database_path = Path(config["database_path"])
    protocol_dir = Path("protocols_trials_extended")  
    model_path = "models/AASIST_CM.pth"     # modify corresponding model path
    wav_path = database_path / "wav"
    output_dir = Path(emb_config["exp_dir"])  # choose export directory 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model_config"], device)

    # modify the output layer to match 13 classes
    num_features = model.out_layer.in_features  # get input size of the output layer
    model.out_layer = torch.nn.Linear(num_features, 13).to(device)  # set new output layer

    # load pre-trained weights, ignoring mismatched layers
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint["out_layer.weight"] = torch.randn(13, num_features)  
    checkpoint["out_layer.bias"] = torch.randn(13) 

    model.load_state_dict(checkpoint, strict=False)  
    print("Model loaded with modified output layer.")

    save_embeddings(protocol_dir, wav_path, model, device, output_dir, config)
    print("Embedding extraction complete!")