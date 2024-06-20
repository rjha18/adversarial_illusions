import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import extract_args, pgd_step, threshold, jpeg, criterion
from dataset_utils import create_dataset
from models import load_model


cfg = extract_args(sys.argv[1])

# Configure Script
cfg.embeddings_input = cfg.embeddings_input.format(cfg.model_flag)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
device = f"cuda:{cfg.gpu_num}" if torch.cuda.is_available() and cfg.gpu_num >= 0 else "cpu"

if cfg.modality == 'vision':
    cfg.epsilon = cfg.epsilon / 255

if type(cfg.epochs) == list:
    max_epochs = max(cfg.epochs)
else:
    max_epochs = cfg.epochs
    cfg.epochs = [cfg.epochs]

assert cfg.number_images % cfg.batch_size == 0

# Instantiate Model
model = load_model(cfg.model_flag, device)

# Load Data
dataset = create_dataset(cfg.dataset_flag, model=model, device=device,
                                    seed=cfg.seed, embs_input=cfg.embeddings_input)
dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

# Create Empty Lists for Logging
X_advs = {e: [] for e in cfg.epochs}
X_inits, gts = [], []                       # Initial images and ground truths
gt_loss, adv_loss = [], []                  # Ground truth and adversarial distances
y_ids, y_origs = [], []                     # Target and input label Ids
end_iter, final, ranks = [], [], []         # State at last iteration

# Create Adversarial Illusionsç
torch.manual_seed(cfg.seed)
for i, (X, Y, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (cfg.number_images // cfg.batch_size):
        break
    X_init = X.clone().detach().cpu()
    X, Y, y_id = X.requires_grad_(True), Y.to(device), y_id.to(device)
    X_max, X_min = threshold(X, cfg.epsilon, cfg.modality, device)

    iters = torch.ones(cfg.batch_size)
    classified = torch.tensor([False] * cfg.batch_size)
    buffer = torch.ones((cfg.batch_size, cfg.buffer_size))

    pbar = tqdm(range(max_epochs))
    lr = cfg.lr
    for j in pbar:
        if hasattr(cfg, 'jpeg') and cfg.jpeg and cfg.modality == 'vision':
            X = jpeg(X).to(device)

        X, embeds, loss = pgd_step(model, X, Y, X_min, X_max, cfg.lr, cfg.modality, device)
        
        if j % cfg.zero_shot_steps == 0:        # Zero-shot classification
            classes = criterion(embeds[:, None, :], dataset.labels[None, :, :], dim=2).argsort(dim=1, descending=True)
            classified = classified | (classes == y_id[:, None])[:, 0].cpu()
            iters[~classified] = j
        buffer[:, j % cfg.buffer_size] = loss.detach()
        change = (buffer.max(dim=1)[0] - buffer.min(dim=1)[0]).min()
        pbar.set_postfix({'loss': loss.clone().detach().cpu(), 'lr': lr, 'change': change.item()})

        if j + 1 in cfg.epochs:
            X_advs[j+1].append(X.detach().cpu().clone())

        if change < cfg.delta:                  # Early Stopping, if desired
            break

        if (j + 1) % cfg.gamma_epochs == 0:
            lr *= 0.9

    # Record batchwise information
    gt_embeddings = model.forward(gt.to(device), cfg.modality, normalize=True).detach().cpu()
    gt_loss.append(criterion(gt_embeddings, Y.cpu(), dim=1))
    adv_loss.append(criterion(embeds.detach().cpu(), Y.cpu(), dim=1))
    end_iter.append(iters)
    X_inits.append(X_init.clone())
    gts.append(gt.cpu().clone())
    y_ids.append(y_id.cpu())
    y_origs.append(y_orig.cpu())
    final.append((classes == y_id[:, None])[:, 0].cpu())
    ranks.append((classes == y_id[:, None]).cpu().int().argmax(axis=1))

    for k, v in X_advs.items():
        np.save(cfg.output_dir + f'x_advs_{k}', np.concatenate(X_advs[k]))
    
    np.save(cfg.output_dir + 'x_inits', np.concatenate(X_inits))
    np.save(cfg.output_dir + 'gts', np.concatenate(gts))
    np.save(cfg.output_dir + 'gt_loss', np.concatenate(gt_loss))
    np.save(cfg.output_dir + 'adv_loss', np.concatenate(adv_loss))
    np.save(cfg.output_dir + 'end_iter', np.concatenate(end_iter))
    np.save(cfg.output_dir + 'y_ids', np.concatenate(y_ids))
    np.save(cfg.output_dir + 'y_origs', np.concatenate(y_origs))
    np.save(cfg.output_dir + 'final', np.concatenate(final))
    np.save(cfg.output_dir + 'ranks', np.concatenate(ranks))
