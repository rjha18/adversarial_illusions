from models import load_model
import torch
import numpy as np

from torch.utils.data import DataLoader

from imagebind.models import imagebind_model

from utils import criterion
from dataset_utils import create_dataset


batch_size = 1
modality = 'audio'
dataset = 'audioset'
model_flag = 'audioclip'
embs_input = f'outputs/embeddings/{dataset}_{model_flag}_embeddings.npy'

device='cpu'

model = load_model(model_flag, device)
model.to(device)
# model.eval()

# Load Data
image_text_dataset = create_dataset(dataset, model=model, device=device, seed=0, embs_input=embs_input)
dataloader = DataLoader(image_text_dataset, batch_size=batch_size, shuffle=True)

ranks = []
losses = []
torch.manual_seed(0)
# model.eval()
for i, (_, y_emb, gt, y, y_o) in enumerate(dataloader):
    if i == 100:
        break
    # print(y_str[0])
    # print(len(image_text_dataset.dataset))
    # print(image_text_dataset.labels.shape)
    # print(image_text_dataset.lab_to_id.__len__())
    # print(image_text_dataset.label_texts.__len__())
    # print(np.unique(image_text_dataset.label_texts).shape)
    # print(np.unique(image_text_dataset.labels, axis=0).shape)
    # input()
    embeds = model.forward(gt, modality=modality)
    # gt_embeds = model.forward(y_str[0], modality='text', normalize=False)
    # print(image_text_dataset.lab_to_id)
    # print(image_text_dataset.lab_to_id[y_str[0]])
    # print(y, y_o)
    # print(gt_embeds.shape, y_emb.shape, image_text_dataset.labels.shape)
    # print(criterion(embeds, y_emb))
    # print(criterion(gt_embeds, embeds))
    # print(criterion(gt_embeds, y_emb))
    # # # print(criterion(gt_embeds, image_text_dataset.labels[y]))
    # # # print(criterion(gt_embeds, image_text_dataset.labels[y_o]))
    # # print(criterion(embeds[:, None, :], image_text_dataset.labels[None, :, :], dim=2).max(dim=1)[0])
    # input()
    # print(np.all(image_text_dataset.labels == gt_embeds, axis=1))
    classes = criterion(embeds[:, None, :], image_text_dataset.labels[None, :, :], dim=2).argsort(dim=1, descending=True)
    ranks.append((classes == y[:, None]).int().argmax(axis=1))
    losses.append(criterion(embeds, y_emb, dim=1).detach().numpy())
r = np.concatenate(ranks)
np.save(f'aud_baseline.npy', r)
print(len(r))
print((r < 1).sum())
print((r < 5).sum())
print(np.mean(losses), np.std(losses)/10)