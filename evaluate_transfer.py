from glob import glob
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

exp = '018'
no_diag = True

# source_models = ['openclip;ViT-B-16;laion2b_s34b_b88k', 'openclip;ViT-B-16;laion400m_e32', 'openclip;ViT-B-16;datacomp_xl_s13b_b90k', 'openclip;ViT-B-16;openai']
# eval_models = ['openclip;ViT-B-16;laion2b_s34b_b88k', 'openclip;ViT-B-16;laion400m_e32', 'openclip;ViT-B-16;datacomp_xl_s13b_b90k', 'openclip;ViT-B-16;openai']
source_models = ['imagebind', 'openclip', 'audioclip_partial', 'openclip_rn50', 'multi']
eval_models = ['imagebind', 'openclip', 'audioclip_partial', 'openclip_rn50']

sms = ['ImageBind', 'OpenCLIP-ViT', 'AudioCLIP', 'OpenCLIP-RN50', 'Ours']
ems = ['ImageBind', 'OpenCLIP-ViT', 'AudioCLIP', 'OpenCLIP-RN50']

a = np.zeros((len(source_models), len(eval_models)))
for i, sm in enumerate(source_models):
    for j, em in enumerate(eval_models):
        means = []

        for f in glob(f'outputs/transfer/matrix/{sm}/{em}/300/correct.npy'):
            means.append(np.load(f).mean())
        a[i, j] = np.max(means) if len(means) > 0 else -1
        print(f'{sm} -> {em}: {max(means)}')


df_cm = pd.DataFrame(a, index = [i for i in sms],
                     columns = [i for i in ems])
plt.figure(figsize = (9,7))
plt.title('Transfer Learning Success Rate ')
plt.yticks(rotation=0, rotation_mode='anchor')

ax = sn.heatmap(df_cm, cmap='OrRd', mask=np.eye(df_cm.shape[0], df_cm.shape[1]) if no_diag else None, cbar=False)
ax.set_xlabel('Transfer Models')
ax.set_ylabel('Source Models')
ax.set_yticklabels(ax.get_yticklabels(), va='center')
# ax.patch.set_facecolor('red')
ax.patch.set_edgecolor('black')
ax.patch.set_hatch('xx')
ax.get_yticklabels()[4].set_fontweight('heavy')



for i in range(len(df_cm.iloc[:, 0]) - 1):
    for j in range(len(df_cm.iloc[i])):
        if i == j:
            continue
        val = df_cm.iloc[i][j]
        col = 'black' if val < 0.5 else 'white'
        ax.text(j + 0.5, i + 0.5, f'{str(val).rstrip('0').rstrip('.')}', ha='center', va='center', color=col, fontweight='ultralight')
for i in range(len(df_cm.iloc[0])):
    ax.text(i + 0.5, 4.5, f'{str(df_cm.iloc[4][i]).rstrip('0').rstrip('.')}', ha='center', va='center', fontweight='heavy', color='white')
plt.tight_layout()
plt.savefig(f'outputs/confusion{exp}{'_nd' if no_diag else ''}.png')