from glob import glob
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

exp = '3'
no_diag = False

# source_models = ['openclip;ViT-B-16;laion2b_s34b_b88k', 'openclip;ViT-B-16;laion400m_e32', 'openclip;ViT-B-16;datacomp_xl_s13b_b90k', 'openclip;ViT-B-16;openai']
# eval_models = ['openclip;ViT-B-16;laion2b_s34b_b88k', 'openclip;ViT-B-16;laion400m_e32', 'openclip;ViT-B-16;datacomp_xl_s13b_b90k', 'openclip;ViT-B-16;openai']
source_models = ['openclip;ViT-B-16;laion2b_s34b_b88k', 'openclip;ViT-H-14;laion2b_s32b_b79k', 'openclip;ViT-L-14;laion2b_s32b_b82k']
eval_models = ['openclip;ViT-B-16;laion2b_s34b_b88k', 'openclip;ViT-H-14;laion2b_s32b_b79k', 'openclip;ViT-L-14;laion2b_s32b_b82k']

loss = np.zeros(7)
for i, sm in enumerate(source_models):
    for j, em in enumerate(eval_models):
        means = {}
        for f in glob(f'outputs/transfer/matrix/{sm}/{em}/*/correct.npy'):
            means[int(f.split('/')[-2])] = np.load(f).mean()
            # print(f.split('/')[-2], means[f.split('/')[-2]])
        s = np.array(list(dict(sorted(means.items())).values()))
        loss += s - s.max()
        print(loss, s.max(), s.argmax())
        # print(f'{sm} -> {em}: {max(means)}')
print(loss.argmax())
