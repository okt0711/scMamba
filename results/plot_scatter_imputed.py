import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import io as sio
from os import makedirs
from os.path import join
from tqdm import tqdm
from ipdb import set_trace as st


# data_name = 'Lau_2020_PNAS'

# data_name = 'Leng_2021_Nat_Neurosci'

# data_name = 'Smajic_2022_brain'

# data_name = 'Zhu_2022_bioRxiv'

data_name = 'Jung_2022_unpublished'

label = sc.read_h5ad(join('/home/gyutaek/hdd2/Data/Process/MAGIC_UMI4000/', data_name, 'data.h5ad'))
label = label.X

np.random.seed(42)
random.seed(42)

path_magic = join('/home/gyutaek/hdd2/Results/MAGIC_UMI4000/', data_name)
path_dca = join('/home/gyutaek/hdd2/Results/DCA_UMI4000/', data_name)
path_hyena = join('/home/gyutaek/hdd2/Results/scHyena_baseline/2024-09-27-07-35-14-850417(imputation_jung_UMI4000)', data_name)
path_mamba = join('/home/gyutaek/hdd2/Results/scMamba/2024-09-25-08-08-52-533110(imputation_jung_5epoch_lr1e-4_256dim_UMI4000)', data_name)

fig_magic_path = join(path_magic, 'fig')
makedirs(fig_magic_path, exist_ok=True)
fig_dca_path = join(path_dca, 'fig')
makedirs(fig_dca_path, exist_ok=True)
fig_hyena_path = join(path_hyena, 'fig')
makedirs(fig_hyena_path, exist_ok=True)
fig_mamba_path = join(path_mamba, 'fig')
makedirs(fig_mamba_path, exist_ok=True)

lims = [0, label.max()]

imputed_hyena = sc.read_h5ad(join(path_hyena, 'imputed_simul.h5ad'))
imputed_hyena.X[imputed_hyena.X < 0] = 0
imputed_hyena = imputed_hyena.X

imputed_mamba = sc.read_h5ad(join(path_mamba, 'imputed_simul.h5ad'))
imputed_mamba.X[imputed_mamba.X < 0] = 0
imputed_mamba = imputed_mamba.X

for n in range(5):
    label_list = np.array([])
    magic_list = np.array([])
    dca_list = np.array([])
    hyena_list = np.array([])
    mamba_list = np.array([])

    imputed_magic = sc.read_h5ad(join(path_magic, 'imputed_simul_' + str(n) +  '.h5ad'))
    imputed_dca = sc.read_h5ad(join(path_dca, 'imputed_simul_' + str(n) +  '.h5ad'))

    imputed_magic.X[imputed_magic.X < 0] = 0
    imputed_dca.X[imputed_dca.X < 0] = 0
    
    imputed_magic = imputed_magic.X
    imputed_dca = imputed_dca.X
    
    for i in tqdm(range(label.shape[0])):
        label_cell = label[i]
        magic_cell = imputed_magic[i]
        dca_cell = imputed_dca[i]
        hyena_cell = imputed_hyena[i]
        mamba_cell = imputed_mamba[i]

        nonzero_idx = np.nonzero(label_cell)[0]
        sub_idx = nonzero_idx[n::5]

        label_cell = label_cell[sub_idx]
        magic_cell = magic_cell[sub_idx]
        dca_cell = dca_cell[sub_idx]
        hyena_cell = hyena_cell[sub_idx]
        mamba_cell = mamba_cell[sub_idx]

        label_list = np.concatenate([label_list, label_cell], axis=0)
        magic_list = np.concatenate([magic_list, magic_cell], axis=0)
        dca_list = np.concatenate([dca_list, dca_cell], axis=0)
        hyena_list = np.concatenate([hyena_list, hyena_cell], axis=0)
        mamba_list = np.concatenate([mamba_list, mamba_cell], axis=0)
    
    # random_idx = np.random.choice(len(label_list), int(len(label_list) * 0.01), replace = False)
    random_idx = np.random.choice(len(label_list), 3000, replace = False)
    label_list = label_list[random_idx]
    magic_list = magic_list[random_idx]
    dca_list = dca_list[random_idx]
    hyena_list = hyena_list[random_idx]
    mamba_list = mamba_list[random_idx]

    fig_magic = sns.jointplot(x=label_list, y=magic_list, kind='kde', fill=True, color='k', cmap='viridis', xlim=lims, ylim=lims)
    plt.plot(lims, lims, color='k')
    fig_magic.savefig(join(fig_magic_path, 'scatter_plot_' + str(n) + '.png'), dpi=300)

    fig_dca = sns.jointplot(x=label_list, y=dca_list, kind='kde', fill=True, color='k', cmap='viridis', xlim=lims, ylim=lims)
    plt.plot(lims, lims, color='k')
    fig_dca.savefig(join(fig_dca_path, 'scatter_plot_' + str(n) + '.png'), dpi=300)

    fig_hyena = sns.jointplot(x=label_list, y=hyena_list, kind='kde', fill=True, color='k', cmap='viridis', xlim=lims, ylim=lims)
    plt.plot(lims, lims, color='k')
    fig_hyena.savefig(join(fig_hyena_path, 'scatter_plot_' + str(n) + '.png'), dpi=300)

    fig_mamba = sns.jointplot(x=label_list, y=mamba_list, kind='kde', fill=True, color='k', cmap='viridis', xlim=lims, ylim=lims)
    plt.plot(lims, lims, color='k')
    fig_mamba.savefig(join(fig_mamba_path, 'scatter_plot_' + str(n) + '.png'), dpi=300)
