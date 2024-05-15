import sys
from pathlib import Path

from tqdm import tqdm
import toml

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import threshold, criterion
from dataset_utils import create_dataset
from models import load_model

# Configure Script
config = toml.load(f'configs/{sys.argv[1]}.toml')['general']

gpu_num = config['gpu_num']
epochs = config['epochs']
batch_size = config['batch_size']
eps = config['epsilon']
zero_shot_steps = config['zero_shot_steps']
lr = config['lr']
eta_min = config['eta_min']
seed = config['seed']
output_dir = config['output_dir']
n_images = config['number_images']
buffer_size = config['buffer_size']
delta = config['delta']
model_flag = config.get('model_flag', 'imagebind')
embs_input = config.get('embeddings_input', output_dir + 'embs.npy')\
                   .format(model_flag)
gamma_epochs = config.get('gamma_epochs', 100)
modality = config.get('modality', 'vision')
dataset_flag = config.get('dataset_flag', 'imagenet')

if modality == 'vision':
    eps = eps / 255

if type(epochs) == list:
    max_epochs = max(epochs)
else:
    max_epochs = epochs
    epochs = [epochs]

Path(output_dir).mkdir(parents=True, exist_ok=True)

device = f"cuda:{gpu_num}" if torch.cuda.is_available() and gpu_num >= 0 else "cpu"
assert n_images % batch_size == 0

# Instantiate Model
model = load_model(model_flag, device)

# Load Data
image_text_dataset = create_dataset(dataset_flag, model=model, device=device, seed=seed, embs_input=embs_input)
dataloader = DataLoader(image_text_dataset, batch_size=batch_size, shuffle=True)

# Create Adversarial Examples
X_advs = {e: [] for e in epochs}
X_inits = []
gts = []
gt_loss = []
adv_loss = []
end_iter = []

# TODO: verify added code
y_ids = []
y_origs = []

final = []
ranks = []
torch.manual_seed(seed)
for i, (X, Y, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (n_images // batch_size):
        break                                                           
    pbar = tqdm(range(max_epochs))
    X_init = X.clone().detach().cpu().requires_grad_(False)
    X, Y, y_id = X.to(device).requires_grad_(True), Y.to(device), y_id.to(device)
    
    # TODO: Revisit clamping
    X_max, X_min = threshold(X, eps, modality, device)

    # TODO: Revisit eta selection
    optimizer = optim.SGD([X], lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               np.arange(gamma_epochs, max_epochs, gamma_epochs),
                                               gamma=0.9)

    iters = torch.ones(batch_size)
    classified = torch.tensor([False] * batch_size)
    buffer = torch.ones((batch_size, buffer_size))

    for j in pbar:
        eta = scheduler.get_last_lr()[0]
        embeds = model.forward(X, modality, normalize=False)
        cton = 1 - criterion(embeds, Y, dim=1).detach().cpu()
        loss = 1 - criterion(embeds, Y, dim=1)
        update = eta * torch.autograd.grad(outputs=loss.mean(), inputs=X)[0].sign()
        X = (X.detach().cpu() - update.detach().cpu()).to(device)
        X = torch.clamp(X, min=X_min, max=X_max).requires_grad_(True)
        
        if j % zero_shot_steps == 0:
            classes = criterion(embeds[:, None, :], image_text_dataset.labels[None, :, :], dim=2).argsort(dim=1, descending=True)
            classified = classified | (classes == y_id[:, None])[:, 0].cpu()
            iters[~classified] = j
        buffer[:, j % buffer_size] = loss.detach()
        change = (buffer.max(dim=1)[0] - buffer.min(dim=1)[0]).min()
        pbar.set_postfix({'loss': cton, 'eta': eta, 'change': change.item()})
        scheduler.step()

        if j + 1 in epochs:
            X_advs[j+1].append(X.detach().cpu().clone())

        if change < delta:
            break

    # Record batchwise information
    gt_embeddings = model.forward(gt.to(device), modality, normalize=True).detach().cpu()
    gt_loss.append(criterion(gt_embeddings, Y.cpu(), dim=1))
    adv_loss.append(criterion(embeds.detach().cpu(), Y.cpu(), dim=1))
    end_iter.append(iters)
    # TODO: verify added code
    y_ids.append(y_id.cpu())
    y_origs.append(y_orig.cpu())
    final.append((classes == y_id[:, None])[:, 0].cpu())
    ranks.append((classes == y_id[:, None]).cpu().int().argmax(axis=1))

    for k, v in X_advs.items():
        np.save(output_dir + f'x_advs_{k}', np.concatenate(X_advs[k]))
        
    np.save(output_dir + 'x_inits', np.concatenate(X_inits))
    np.save(output_dir + 'gts', np.concatenate(gts))
    np.save(output_dir + 'gt_loss', np.concatenate(gt_loss))
    np.save(output_dir + 'adv_loss', np.concatenate(adv_loss))
    np.save(output_dir + 'end_iter', np.concatenate(end_iter))
    
    # TODO: verify added code
    np.save(output_dir + 'y_ids', np.concatenate(y_ids))
    np.save(output_dir + 'y_origs', np.concatenate(y_origs))
    np.save(output_dir + 'final', np.concatenate(final))
    np.save(output_dir + 'ranks', np.concatenate(ranks))
