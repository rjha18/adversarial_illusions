import os
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets

import librosa
from tqdm import tqdm
import imagebind.data as data

DATA_PATH = {
    'imagenet': '../data/imagenet/',
    'audiocaps': '../data/AudioCaps/',
    'audioset': '../data/AudioCaps/',
}


class WrappedImageNetDataset(Dataset):
    def __init__(
        self, dataset, labels, template, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=250,
        embedding_override=False
    ):
        self.dataset = dataset
        self.seed = seed
        self.template = template
        self.model = model
        np.random.seed(seed=self.seed)
        self.mapping = mapping if mapping is not None else np.random.permutation(len(dataset))
        self.device = device
        self.embs_file = embs_input
        self.labels = self.get_embeddings(labels, embedding_batch_size, embedding_override)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y_orig_id = self.dataset[idx]
        gt, y_id = self.dataset[self.mapping[idx]]
        y = self.labels[y_id].to(self.device)
        return torch.squeeze(x), torch.squeeze(y), torch.squeeze(gt), y_id, y_orig_id

    def get_embeddings(self, labels, batch_size=250, device_override=False):
        if self.embs_file is not None and os.path.isfile(self.embs_file):
            return torch.tensor(np.load(self.embs_file)).to(self.device)

        labs = np.stack([self.template.format(labels[i].split(',')[0]) for i in range(len(labels))])
        embs = []

        for i in tqdm(range(len(labs) // batch_size)):
            batch = labs[i*batch_size:(i+1)*batch_size]
            with torch.no_grad():
                embs.append(self.model.cpu().forward(batch, 'text', normalize=False))

        if not device_override:
            self.model.to(self.device)

        if self.embs_file is not None:
            folder_path = os.path.dirname(self.embs_file)
            os.makedirs(folder_path, exist_ok=True)
            np.save(self.embs_file, torch.cat(embs).to(self.device).cpu())
        return torch.concatenate(embs).to(self.device)


class WrappedAudioCapsDataset(Dataset):
    def __init__(
        self, dataset, model,
        mapping=None, device='cpu', seed=0,
        embs_input=None, embedding_batch_size=250,
        embedding_override=False
    ):
        self.dataset = dataset
        self.seed = seed
        self.model = model
        np.random.seed(seed=self.seed)
        self.mapping = mapping if mapping is not None else np.random.permutation(len(dataset))
        self.device = device
        self.embs_file = embs_input

        self.label_texts = list(dict.fromkeys([y for _, y in dataset]))
        self.lab_to_id = {l: i for i, l in enumerate(self.label_texts)}
        self.labels = self.get_embeddings(self.label_texts, embedding_batch_size, embedding_override)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y_orig = self.dataset[idx]
        gt, y_str = self.dataset[self.mapping[idx]]
        y_orig_id, y_str_id = self.lab_to_id[y_orig], self.lab_to_id[y_str]
        y = self.labels[y_str_id].to(self.device)

        if self.model.flag == 'imagebind':
            x = torch.squeeze(x)[:, None, :, :]
            gt = torch.squeeze(gt)[:, None, :, :]
        x = (0.0001 * torch.randn_like(x)) + x.detach()
        return torch.squeeze(x), torch.squeeze(y), gt, y_str_id, y_orig_id

    def get_embeddings(self, labels, batch_size=250, device_override=False):
        if self.embs_file is not None and os.path.isfile(self.embs_file):
            return torch.tensor(np.load(self.embs_file)).to(self.device)

        embs = []
        for i in range((len(labels) // batch_size) + 1):
            batch = labels[i*batch_size:(i+1)*batch_size]
            with torch.no_grad():
                embs.append(self.model.cpu().forward(batch, 'text', normalize=False))
        if not device_override:
            self.model.to(self.device)
        if self.embs_file is not None:
            np.save(self.embs_file, torch.concatenate(embs).to(self.device).cpu())
        return torch.concatenate(embs).to(self.device)


class AudioDataset(Dataset):
    def __init__(self, audio_dir, split_file, extension='wav', device='cpu', model_flag='imagebind'):
        self.audio_files = glob.glob(f'{audio_dir}*.{extension}')
        self.split = pd.read_csv(split_file, index_col='youtube_id')[['caption']]
        self.device = device
        self.model_flag = model_flag
        assert len(self.audio_files) > 0
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        path = self.audio_files[idx]
        if self.model_flag == 'imagebind':
            X = data.load_and_transform_audio_data([path], self.device)
        elif self.model_flag == 'audioclip':
            X = librosa.load(path, sr=44100, dtype=np.float32)[0]
            X = torch.tensor(X).to(self.device)
        y = self.split.loc[self.get_id(path)].iloc[-1].item()
        return X, y

    def get_id(self, path):
        return path.split('/')[-1].split('.')[0]


def imagenet_loader(path, model, device='cpu'):
    if model.flag == 'imagebind' or model.flag == 'audioclip':
        return data.load_and_transform_vision_data([path], device)
    elif model.flag == 'openclip':
        image_outputs = []
        with open(path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = model.preprocess(image).to(device)
        image_outputs.append(image)
        return torch.stack(image_outputs, dim=0)
    else:
        raise NotImplementedError()


def create_dataset(
    dataset_flag, model=None, mapping=None, device='cpu', embs_input=None, seed=0
):
    assert model is not None
    if dataset_flag == 'imagenet':
        loader = lambda p: imagenet_loader(p, model, device)
        imagenet = datasets.ImageNet(DATA_PATH[dataset_flag], split='val', loader=loader)
        with open(DATA_PATH[dataset_flag] + 'imagenet1000_clsidx_to_labels.txt') as f:
            labels = eval(f.read().replace('\n', ''))
        template = "A photo of a {}."
        return WrappedImageNetDataset(imagenet, labels, template, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'audiocaps':
        audiocaps = AudioDataset(DATA_PATH[dataset_flag] + 'raw/',
                              DATA_PATH[dataset_flag] + 'splits/retrieval_test.csv',
                              'wav',
                              model_flag=model.flag)
        return WrappedAudioCapsDataset(audiocaps, model, mapping, device, seed, embs_input)
    elif dataset_flag == 'audioset':
        audioset = AudioDataset(DATA_PATH[dataset_flag] + 'raw/',
                                DATA_PATH[dataset_flag] + 'splits/classification_test.csv',
                                'wav',
                                model_flag=model.flag)
        return WrappedAudioCapsDataset(audioset, model, mapping, device, seed, embs_input)
    else:
        raise NotImplementedError
