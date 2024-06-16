import sys
from pathlib import Path

from tqdm import tqdm
import toml

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import threshold, criterion, unnorm
from dataset_utils import create_dataset
from models import load_model

# Configure Script
config = toml.load(f'configs/{sys.argv[1]}.toml')['general']

gpu_nums = config['gpu_nums']
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
model_flags = config.get('model_flags', ['imagebind'])
gamma_epochs = config.get('gamma_epochs', 100)
modality = config.get('modality', 'vision')
dataset_flag = config.get('dataset_flag', 'imagenet')
target_model_flag = config.get('target_model_flag', None)
target_device = config.get('target_device', None)

assert (target_model_flag is None) == (target_device is None)

if modality == 'vision':
    eps = eps / 255

if type(epochs) == list:
    max_epochs = max(epochs)
else:
    max_epochs = epochs
    epochs = [epochs]

Path(output_dir).mkdir(parents=True, exist_ok=True)

devices = [f"cuda:{g}" if torch.cuda.is_available() and g >= 0 else "cpu" for g in gpu_nums]
assert n_images % batch_size == 0
embeddings = [np.load(f'outputs/embeddings/{dataset_flag}_{f}_embeddings.npy') for f in model_flags]    


# Instantiate Model
models = [(
    load_model(f, devices[i % len(devices)]),
    torch.tensor(embeddings[i % len(devices)]).to(devices[i % len(devices)]),
    devices[i % len(devices)])
for i, f in enumerate(model_flags)]

embs_input = f'outputs/embeddings/{dataset_flag}_{model_flags[0]}_embeddings.npy'    

# Load Data
dataset = create_dataset(dataset_flag, model=models[0][0], device=models[0][2], seed=seed, embs_input=embs_input)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

# if (target_model_flag is not None) and (target_device is not None):
#     target_embs = torch.tensor(np.load(f'outputs/embeddings/{dataset_flag}_{target_model_flag}_embeddings.npy')).to(target_device)
#     target_model = load_model(target_model_flag, target_device)

final = []
torch.manual_seed(seed)
for i, (X, _, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (n_images // batch_size):
        break
    
    X_init = X.clone().detach().cpu().requires_grad_(False)
    pbar = tqdm(range(max_epochs))
    optimizer = optim.SGD([X], lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               np.arange(gamma_epochs, max_epochs, gamma_epochs),
                                               gamma=0.9)

    iters = torch.ones(batch_size)
    classified = torch.tensor([False] * batch_size)
    buffer = torch.ones((batch_size, buffer_size))
    X_max, X_min = threshold(X, eps, modality, models[0][2])

    for j in pbar:
        loss = torch.tensor([0.0] * batch_size)
        right = torch.tensor(0)
        cton = []
        for m, l, d in models:
            Y = l[y_id]
            X_m, Y_m = X.to(d).requires_grad_(True), Y.to(d)
        
            eta = scheduler.get_last_lr()[0]
            embeds = m.forward(X_m, modality, normalize=False)
            cton.append((1 - criterion(embeds, Y_m, dim=1).detach().cpu()).mean().item())
            loss += 1 - criterion(embeds, Y_m, dim=1).cpu()
            update = eta * torch.autograd.grad(outputs=loss.mean() / len(models), inputs=X_m)[0].sign()
            X = (X.detach().cpu() - update.detach().cpu()).to(models[0][2])
            X = torch.clamp(X, min=X_min, max=X_max).requires_grad_(True)
            pbar.set_postfix({'dist': (unnorm(X_init.cpu()) - unnorm(X.cpu())).abs().max().item()})
        # if (target_model_flag is not None) and (target_device is not None):
        #     embeds = target_model.forward(X.tos('cuda:0'), modality, normalize=False)
        #     classes = criterion(embeds[:, None, :], target_embs[None, :, :], dim=2).argsort(dim=1, descending=True).cpu()
        #     classified = classified | (classes == y_id[:, None])[:, 0].cpu()
        #     right += (classified | (classes == y_id[:, None])[:, 0].cpu()).sum()
        #     iters[~classified] = j
        #     cton.append((1 - criterion(embeds, target_embs[y_id], dim=1).detach().cpu()).mean().item())
        #     pbar.set_postfix({'loss': cton, 'r': right.item()})
        scheduler.step()

        if j + 1 in epochs:
            X_advs[j+1].append(X.detach().cpu().clone())

    # Record batchwise information
    # if (target_model_flag is not None) and (target_device is not None):
    #     gt_embeddings = target_model.forward(gt.to(target_device), modality, normalize=True).detach().cpu()
    #     adv_loss.append(criterion(embeds.detach().cpu(), Y.cpu(), dim=1))
    #     end_iter.append(iters)
    #     # TODO: verify added code
    #     y_origs.append(y_orig.cpu())
    #     final.append((classes == y_id[:, None])[:, 0].cpu())
   
    #     gts.append(gt.cpu().clone())
    #     gt_loss.append(criterion(gt_embeddings, Y.cpu(), dim=1))
    
    y_ids.append(y_id.cpu())
    X_inits.append(X_init.clone())

    for k, v in X_advs.items():
        np.save(output_dir + f'x_advs_{k}', np.concatenate(X_advs[k]))
        
    # np.save(output_dir + 'x_inits', np.concatenate(X_inits))
    # np.save(output_dir + 'gts', np.concatenate(gts))
    # np.save(output_dir + 'gt_loss', np.concatenate(gt_loss))
    # np.save(output_dir + 'adv_loss', np.concatenate(adv_loss))
    # np.save(output_dir + 'end_iter', np.concatenate(end_iter))
    
    # # TODO: verify added code
    # np.save(output_dir + 'y_ids', np.concatenate(y_ids))
    # np.save(output_dir + 'y_origs', np.concatenate(y_origs))
    # np.save(output_dir + 'final', np.concatenate(final))
