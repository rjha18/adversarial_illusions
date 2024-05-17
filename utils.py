import torch
import torch.nn.functional as F


criterion = F.cosine_similarity

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

def norm_audioo(tensor, mean=IMG_MEAN, std=IMG_STD):
    m = -4.268
    s = 9.138
    return ((tensor - m) / s)
        
def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def di(image):
    display(transform(unnorm(image.data.detach().cpu())))


def threshold(X, eps, modality, device):
    if modality == 'vision':
        X_unnorm = unnorm(X.data)
        X_max, X_min = norm(torch.clamp(X_unnorm+eps, min=0, max=1)), norm(torch.clamp(X_unnorm-eps, min=0, max=1))
    elif modality == 'thermal':
        X_unnorm = unnorm(X.data, THERMAL_MEAN, THERMAL_STD)
        X_max, X_min = norm(torch.clamp(X_unnorm+eps, min=0, max=1), THERMAL_MEAN, THERMAL_STD), norm(torch.clamp(X_unnorm-eps, min=0, max=1), THERMAL_MEAN, THERMAL_STD)
    elif modality == 'audio':
        X_max, X_min = X + eps, X - eps
    return X_max.to(device), X_min.to(device)
