from argparse import Namespace
import toml
import torch

criterion = torch.nn.functional.cosine_similarity

IMG_MEAN=(0.48145466, 0.4578275, 0.40821073)
IMG_STD=(0.26862954, 0.26130258, 0.27577711)
THERMAL_MEAN=(0.2989 * IMG_MEAN[0]) + (0.5870 * IMG_MEAN[1]) + (0.1140 * IMG_MEAN[2])
THERMAL_STD=(0.2989 * IMG_STD[0]) + (0.5870 * IMG_STD[1]) + (0.1140 * IMG_STD[2])

def unnorm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() * std) + mean).to(device)

def norm(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    if type(mean) != float:
        mean = torch.tensor(mean)[None, :, None, None]
        std = torch.tensor(std)[None, :, None, None]
    return ((tensor.clone().cpu() - mean) / std).to(device)


def unnorm_audio(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(-4.268)
    s = torch.tensor(9.138)
    return ((tensor.clone().cpu() * s) + m).to(device)

def norm_audio(tensor, mean=IMG_MEAN, std=IMG_STD):
    device = tensor.get_device() if tensor.get_device() > 0 else 'cpu'
    m = torch.tensor(-4.268)
    s = torch.tensor(9.138)
    return ((tensor.clone().cpu() - m) / s).to(device)

def threshold(X, eps, modality, device):
    if modality == 'vision':
        X_unnorm = unnorm(X.data)
        X_max, X_min = norm(torch.clamp(X_unnorm+eps, min=0, max=1)), norm(torch.clamp(X_unnorm-eps, min=0, max=1))
    elif modality == 'thermal':
        X_max, X_min = torch.clamp(X+eps, min=0, max=1), torch.clamp(X-eps, min=0, max=1)
    elif modality == 'audio':
        X_max, X_min = X + eps, X - eps
    return X_max.to(device), X_min.to(device)

def extract_args(exp_name):
    return Namespace(**toml.load(f'configs/{exp_name}.toml')['general'])
