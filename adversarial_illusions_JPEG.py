import sys
from pathlib import Path

from tqdm import tqdm
import toml

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import norm, unnorm, criterion
from dataset_utils import create_dataset
from models import load_model

sys.path.insert(0,'/home/eugene/DiffJPEG')  # Adds the DiffJPEG directory to sys.path
from DiffJPEG import DiffJPEG

# Configure Script
config = toml.load(f'configs/{sys.argv[1]}.toml')['general']

gpu_num = config['gpu_num']
max_epochs = config['max_epochs']
batch_size = config['batch_size']
eps = config['epsilon'] / 255
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

Path(output_dir).mkdir(parents=True, exist_ok=True)

device = f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
assert n_images % batch_size == 0

# Instantiate Model
model = load_model(model_flag, device)
modality = 'vision'

# Load Data
image_text_dataset = create_dataset('imagenet', model=model, device=device, seed=seed, embs_input=embs_input)
dataloader = DataLoader(image_text_dataset, batch_size=batch_size, shuffle=True)


jpeg = DiffJPEG(224, 224, differentiable=True, quality=80)

# Create Adversarial Examples
X_advs = []
X_inits = []
gts = []
gt_loss = []
adv_loss = []
end_iter = []

# TODO: verify added code
y_ids = []
y_origs = []

final = []
torch.manual_seed(seed)
for i, (X, Y, gt, y_id, y_orig) in enumerate(dataloader):
    if i >= (n_images // batch_size):
        break                                                           
    pbar = tqdm(range(max_epochs))
    X_init = X.clone().detach().cpu().requires_grad_(False)
    X, Y, y_id = X.to(device).requires_grad_(True), Y.to(device), y_id.to(device)
    
    # TODO: Revisit clamping
    X_unnorm = unnorm(X.data)
    X_max, X_min = torch.clamp(X_unnorm+eps, min=0, max=1), torch.clamp(X_unnorm-eps, min=0, max=1)
    X_max, X_min = norm(X_max).to(device), norm(X_min).to(device)

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
        img_tensor=unnorm(X).squeeze(0).cpu()
        img_tensor.retain_grad()
        img_jpeg = jpeg(img_tensor).to(device)
    
        embeds = model.forward(norm(img_jpeg), modality, normalize=False)
        # embeds = model.forward(X, modality, normalize=False)
    
        cton = 1 - criterion(embeds, Y, dim=1).detach().cpu()
        loss = 1 - criterion(embeds, Y, dim=1)
        # print(loss)
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

        if change < delta:
            break

    # Record batchwise information
    gt_embeddings = model.forward(gt.to(device), modality, normalize=True).detach().cpu()
    X_advs.append(X.detach().cpu().clone())
    X_inits.append(X_init.clone())
    gts.append(gt.cpu().clone())
    gt_loss.append(criterion(gt_embeddings, Y.cpu(), dim=1))
    adv_loss.append(criterion(embeds.detach().cpu(), Y.cpu(), dim=1))
    end_iter.append(iters)
    print((classes == y_id[:, None])[:, 0])
    # TODO: verify added code
    y_ids.append(y_id.cpu())
    y_origs.append(y_orig.cpu())
    final.append((classes == y_id[:, None])[:, 0].cpu())

    np.save(output_dir + 'x_advs', np.concatenate(X_advs))
    np.save(output_dir + 'x_inits', np.concatenate(X_inits))
    np.save(output_dir + 'gts', np.concatenate(gts))
    np.save(output_dir + 'gt_loss', np.concatenate(gt_loss))
    np.save(output_dir + 'adv_loss', np.concatenate(adv_loss))
    np.save(output_dir + 'end_iter', np.concatenate(end_iter))
    
    # TODO: verify added code
    np.save(output_dir + 'y_ids', np.concatenate(y_ids))
    np.save(output_dir + 'y_origs', np.concatenate(y_origs))
    np.save(output_dir + 'final', np.concatenate(final))
