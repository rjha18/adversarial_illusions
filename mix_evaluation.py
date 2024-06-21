import sys
from pathlib import Path

import toml

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import criterion
from dataset_utils import create_dataset
from models import load_model

# Configure Script
config = toml.load(f'configs/{sys.argv[1]}.toml')['general']

input_ims_file = config['input_images_file']
target_model_flag = config['target_model_flag']

gpu_num = config['gpu_num']
batch_size = config['batch_size']
seed = config['seed']
output_dir = config['output_dir']
n_images = config['number_images']

embs_input = config.get('embeddings_input', output_dir + 'embs.npy')\
                   .format(target_model_flag)

Path(output_dir).mkdir(parents=True, exist_ok=True)

device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
assert n_images % batch_size == 0

# Instantiate Model
model = load_model(target_model_flag, device)
modality = 'vision'

# Load Data
dataset = create_dataset('imagenet', model=model, device=device, seed=seed, embs_input=embs_input)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

examples = torch.tensor(np.load(input_ims_file))

torch.manual_seed(seed)
classifieds = []
dists = []
for i, (_, _, _, y_id, _) in enumerate(dataloader):
    if i >= 25:
        break             

    y_id = y_id.to(device)
    idx = np.arange(4 * i, 4 * (i+1))

    embeds = model.forward(examples[idx].to(device), modality, normalize=False).detach().to(device)

    classified = torch.tensor([False] * batch_size)
    dist = criterion(embeds[:, None, :], dataset.labels[None, :, :], dim=2)
    max_dists, classes = dist.sort(dim=1, descending=True)
    classifieds.append(classified | (classes == y_id[:, None])[:, 0].cpu())
    dists.append(max_dists[:, 0].cpu())

print(np.concatenate(classifieds).mean(), np.concatenate(dists).mean())

np.save(output_dir + 'correct', np.concatenate(classifieds))
np.save(output_dir + 'dists', np.concatenate(dists))
