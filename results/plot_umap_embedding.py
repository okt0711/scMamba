import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import makedirs
from os.path import join


def plot_umap(umap_data, names, colormap, groups, fig_name, legend=False):
    fig, ax = plt.subplots(1, figsize=(8, 7))

    for name in names:
        group = umap_data[groups == name, :]
        ax.scatter(group[:, 0], group[:, 1], linewidths=0, s=5, marker='o', label=name, color=colormap[names.index(name)])
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_xlim([min(umap_data[:, 0]) - 2, max(umap_data[:, 0]) + 2])
    ax.set_ylim([min(umap_data[:, 1]) - 2, max(umap_data[:, 1]) + 2])
    if legend:
        Line, Label = ax.get_legend_handles_labels()
        fig.legend(Line, Label, loc='upper right')

    fig.savefig(fig_name, dpi=300)
    plt.close()

##### Data name #####
data_name = 'Lau_2020_PNAS'
# data_name = 'Leng_2021_Nat_Neurosci'
# data_name = 'Smajic_2022_brain'
# data_name = 'Zhu_2022_bioRxiv'
# data_name = 'Jung_2022_unpublished'

##### Output path #####
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-05-24-08-25-24-135401(pretrain_2epoch_lr1e-3_256dim)', data_name)
embedding_file = sc.read_h5ad(join(path_output, 'embedding.h5ad'))

filter_cells = ['Double_negative_Neuron', 'Double_positive_Neuron', 'GNLY_CD44_myeloid_sub1']
celltype_bools = np.invert([x in filter_cells for x in embedding_file.obs['celltype']])
embedding_file = embedding_file[celltype_bools].copy()
embedding = embedding_file.X

path_fig = join(path_output, 'fig_v2')
path_fig_label = join(path_output, 'fig_label')
makedirs(path_fig, exist_ok=True)
makedirs(path_fig_label, exist_ok=True)

embedding = StandardScaler().fit_transform(embedding)

n_neighbors = 100
min_dist = 0.1
umap.UMAP()
umap_data = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='correlation', random_state=0).fit_transform(embedding)

names_celltype = ['Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Excitatory', 'Inhibitory', 'Endothelial', 'Pericyte']
groups_celltype = embedding_file.obs['celltype']
colormap_celltype = np.array([[62, 158, 100], [255, 153, 0], [217, 49, 55], [83, 55, 125], [45, 140, 184], [184, 100, 204], [196, 236, 4], [240, 222, 0]], dtype=np.float32) / 255.0
colormap_celltype = np.concatenate([colormap_celltype, np.ones((len(names_celltype), 1), dtype=np.float32)], axis=1)
fig_name_celltype = join(path_fig, f'umap_celltype.png')
fig_name_celltype_label = join(path_fig_label, f'umap_celltype.png')
plot_umap(umap_data, names_celltype, colormap_celltype, groups_celltype, fig_name_celltype)
plot_umap(umap_data, names_celltype, colormap_celltype, groups_celltype, fig_name_celltype_label, legend=True)

# names_donor = np.unique(list(embedding_file.obs['donor'])).tolist()
# groups_donor = embedding_file.obs['donor']
# colormap_donor = np.linspace(1, 0, 7)[:len(names_donor)]
# colormap_donor = cm.rainbow(colormap_donor)
# fig_name_donor = join(path_fig, 'umap_donor.png')
# fig_name_donor_label = join(path_fig_label, 'umap_donor.png')
# plot_umap(umap_data, names_donor, colormap_donor, groups_donor, fig_name_donor)
# plot_umap(umap_data, names_donor, colormap_donor, groups_donor, fig_name_donor_label, legend=True)
