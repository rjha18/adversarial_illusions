import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import extract_args, pgd_step, threshold, criterion
from dataset_utils import create_dataset
from models import load_model


cfg = extract_args(sys.argv[1])

# Configure Script
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
devices = [f"cuda:{g}" if torch.cuda.is_available() and g >= 0 else "cpu" for g in cfg.gpu_nums]
embeddings = [np.load(f'outputs/embeddings/{cfg.dataset_flag}_{f}_embeddings.npy') for f in cfg.model_flags]    


if cfg.modality == 'vision':
    cfg.epsilon = cfg.epsilon / 255

if type(cfg.epochs) == list:
    max_epochs = max(cfg.epochs)
else:
    max_epochs = cfg.epochs
    cfg.epochs = [cfg.epochs]


assert cfg.number_images % cfg.batch_size == 0


# Instantiate Model
models = [(
    load_model(f, devices[i % len(devices)]),
    torch.tensor(embeddings[i % len(devices)]).to(devices[i % len(devices)]),
    devices[i % len(devices)])
for i, f in enumerate(cfg.model_flags)]
target_model = load_model(cfg.target_model_flag, cfg.target_device)

embs_input = f'outputs/embeddings/{cfg.dataset_flag}_{cfg.model_flags[0]}_embeddings.npy'    

# Load Data
dataset = create_dataset(cfg.dataset_flag, model=models[0][0], device=models[0][2], seed=cfg.seed, embs_input=embs_input)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# Create Empty Lists for Logging
X_advs = {e: [] for e in cfg.epochs}
X_inits, gts = [], []                       # Initial images and ground truths
gt_loss, adv_loss = [], []                  # Ground truth and adversarial distances
y_ids, y_origs = [], []                     # Target and input label Ids

# Create Adversarial Examples
torch.manual_seed(cfg.seed)
for i, (X, Y, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (cfg.number_images // cfg.batch_size):
        break
    X_init = X.clone().detach().cpu()
    X_max, X_min = threshold(X, cfg.epsilon, cfg.modality, models[0][2])

    pbar = tqdm(range(max_epochs))
    for j in pbar:
        total_loss = torch.tensor([0.0] * cfg.batch_size)
        for m, l, d in models:
            Y = l[y_id]
            X_m, Y_m = X.to(d).requires_grad_(True), Y.to(d)

            X, embeds, loss = pgd_step(m, X_m, Y_m, X_min, X_max, cfg.lr, cfg.modality, models[0][2])
        
            total_loss += loss.clone().detach().cpu()
        pbar.set_postfix({'loss': total_loss / len(models), 'lr': cfg.lr})

        if j + 1 in cfg.epochs:
            X_advs[j+1].append(X.detach().cpu().clone())
        
        if j + 1 % cfg.gamma_epochs == 0:
            cfg.lr = 0.9 * cfg.lr

    # Record batchwise information
    if (cfg.target_model_flag is not None) and (cfg.target_device is not None):
        gt_embeddings = target_model.forward(gt.to(cfg.target_device), cfg.modality, normalize=True).detach().cpu()
        adv_loss.append(criterion(embeds.detach().cpu(), Y.cpu(), dim=1))
        y_origs.append(y_orig.cpu())
        gts.append(gt.cpu().clone())
        gt_loss.append(criterion(gt_embeddings, Y.cpu(), dim=1))
    
    y_ids.append(y_id.cpu())
    X_inits.append(X_init.clone())

    for k, v in X_advs.items():
        np.save(cfg.output_dir + f'x_advs_{k}', np.concatenate(X_advs[k]))
        
    np.save(cfg.output_dir + 'x_inits', np.concatenate(X_inits))
    np.save(cfg.output_dir + 'gts', np.concatenate(gts))
    np.save(cfg.output_dir + 'gt_loss', np.concatenate(gt_loss))
    np.save(cfg.output_dir + 'adv_loss', np.concatenate(adv_loss))
    np.save(cfg.output_dir + 'y_ids', np.concatenate(y_ids))
    np.save(cfg.output_dir + 'y_origs', np.concatenate(y_origs))
