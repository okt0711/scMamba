import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import makedirs
from os.path import join


def customize_axes(ax, name):
    ax.set_title(name, size=15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP 1', size=12)
    ax.set_ylabel('UMAP 2', size=12)

##### Data name #####
# data_name = 'Lau_2020_PNAS'
# data_name = 'Leng_2021_Nat_Neurosci'
# data_name = 'Smajic_2022_brain'
# data_name = 'Zhu_2022_bioRxiv'
data_name = 'Jung_2022_unpublished'

##### Output path #####
# path_output = join('/home/gyutaek/hdd2/Data/Process/MAGIC', data_name)
# imputed = sc.read_h5ad(join(path_output, 'data.h5ad'))

# path_output = join('/home/gyutaek/hdd2/Results/MAGIC', data_name)
# path_output = join('/home/gyutaek/hdd2/Results/DCA', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scHyena_baseline/2024-09-27-07-35-14-850417(imputation_jung_UMI4000)', data_name)
# path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-09-25-08-08-52-533110(imputation_jung_5epoch_lr1e-4_256dim_UMI4000)', data_name)
imputed = sc.read_h5ad(join(path_output, 'imputed.h5ad'))
imputed.X[imputed.X < 0] = 0

filter_cells = ['Double_negative_Neuron', 'Double_positive_Neuron', 'GNLY_CD44_myeloid_sub1']
celltype_bools = np.invert([x in filter_cells for x in imputed.obs['celltype']])
imputed = imputed[celltype_bools].copy()

path_fig = join(path_output, 'fig')
makedirs(path_fig, exist_ok=True)

n_neighbors = 15
min_dist = 0.5
sc.pp.pca(imputed, n_comps=30)
sc.pp.neighbors(imputed, n_neighbors=n_neighbors, n_pcs=30)
sc.tl.umap(imputed, min_dist=min_dist)

batch_map_celltype = {cell: name for cell, name in zip(imputed.obs.index, imputed.obs['celltype'])}

embedding_celltype = pd.DataFrame(imputed.obsm['X_umap'], index=imputed.obs.index)
embedding_celltype['Batch'] = [batch_map_celltype[cell] for cell in embedding_celltype.index]
embedding_celltype = embedding_celltype.groupby('Batch')

f_celltype, ax_celltype = plt.subplots(1, figsize=(8, 7))
names_celltype = ['Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Excitatory', 'Inhibitory', 'Endothelial', 'Pericyte']
colormap_celltype = np.array([[62, 158, 100], [255, 153, 0], [217, 49, 55], [83, 55, 125], [45, 140, 184], [184, 100, 204], [196, 236, 4], [240, 222, 0]], dtype=np.float32) / 255.0
colormap_celltype = np.concatenate([colormap_celltype, np.ones((len(names_celltype), 1), dtype=np.float32)], axis=1)

for name, c in zip(names_celltype, colormap_celltype):
    group = embedding_celltype.get_group(name)
    ax_celltype.scatter(group[0], group[1], linewidths=0, s=5, marker='o', color=c, label=name)

f_celltype.savefig(join(path_fig, 'umap_celltype.png'), dpi=300)

batch_map_donor = {cell: name for cell, name in zip(imputed.obs.index, imputed.obs['donor'])}

embedding_donor = pd.DataFrame(imputed.obsm['X_umap'], index=imputed.obs.index)
embedding_donor['Batch'] = [batch_map_donor[cell] for cell in embedding_donor.index]
embedding_donor = embedding_donor.groupby('Batch')

f_donor, ax_donor = plt.subplots(1, figsize=(8, 7))
names_donor = np.unique(list(batch_map_donor.values()))
colormap_donor = np.linspace(1, 0, 7)[:len(names_donor)]
colormap_donor = cm.rainbow(colormap_donor)

for name, c in zip(names_donor, colormap_donor):
    group = embedding_donor.get_group(name)
    ax_donor.scatter(group[0], group[1], linewidths=0, s=5, marker='o', color=c, label=name)

f_donor.savefig(join(path_fig, 'umap_donor.png'), dpi=300)
