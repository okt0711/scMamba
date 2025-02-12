import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import makedirs
from os.path import join


def plot_umap(umap_data, names, colormap, groups, fig_name, subtype_list, legend=False):
    if legend:
        fig, ax = plt.subplots(1, figsize=(24, 21))
    else:
        fig, ax = plt.subplots(1, figsize=(8, 7))

    for name in names:
        if name not in subtype_list:
            continue
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
# data_name = 'Lau_2020_PNAS'
# data_name = 'Leng_2021_Nat_Neurosci'
# data_name = 'Smajic_2022_brain'
# data_name = 'Zhu_2022_bioRxiv'
data_name = 'Jung_2022_unpublished'

names_celltype = []
for subtype in ['APOE_astrocyte', 'CHII3L1_astrocyte', 'DPP10_astrocyte', 'GRM3_astrocyte', 'Intermediate']:
    names_celltype.append('_'.join(['Astrocyte', subtype]))

for subtype in ['Homeostatic', 'Homeostatic_defficent', 'InflammatoryI', 'InflammatoryII', 'Lipid_processing', 'Phago_Infla_intermediate', 'Phagocytic', 'Ribosomal_genesis']:
    names_celltype.append('_'.join(['Microglia', subtype]))

for subtype in ['CAMK2D_Oligo', 'OPALIN_FRY_Oligo', 'OPALIN_Oligo', 'OPALIN_high_Oligo', 'RBFOX1_Oligo', 'RBFOX1_high_Oligo', 'highMT', 'intermediate']:
    names_celltype.append('_'.join(['Oligodendrocyte', subtype]))

names_celltype.append('OPC')

for subtype in ['CALB', 'L2-4_Lamp5', 'L2-4_SYT2', 'L2_3', 'L4/5_RORB_GABRG1', 'L4/5_RORB_LINC02196', 'L4/5_RORB_PCP4', 'L4/5_RORB_PLCH1_MME', 'L4/5_RORB_RPRM', 'L4_RORB_COL5A2_PLCH1',
                'L5/6_NFIA_THEMIS', 'L5/6_NXPH2', 'L5_ET_SYT2', 'L6_HPSE2_NR4A2_NTNG2', 'L6b_PCSK5_SEMA3D', 'L6b_PCSK5_SULF1', 'NRGN', 'RELN_CHD7', 'SOX6', 'high_MT']:
    names_celltype.append('_'.join(['Excitatory', subtype]))

for subtype in ['ALCAM_TRPM3', 'CUX2_MSR1', 'ENOX2_SPHKAP', 'FBN2_EPB41L4A', 'GPC5_RIT2', 'LAMP5_CA13', 'LAMP5_NRG1', 'LAMP5_RELN', 'PAX6_CA4',
                'PTPRK_FAM19A1', 'PVALB_CA8', 'PVALB_SULF1_HTR4', 'RYR3_TSHZ2', 'RYR3_TSHZ2_VIP_THSD7B', 'SGCD_PDE3A', 'SN1', 'SN2',
                'SORCS1_TTN', 'SST_MAFB', 'SST_NPY', 'VIP_ABI3BP_THSD7B', 'VIP_CLSTN2', 'VIP_THSD7B', 'VIP_TSHZ2', 'high_MT']:
    names_celltype.append('_'.join(['Inhibitory', subtype]))

if data_name == 'Leng_2021_Nat_Neurosci':
    names_celltype.append('Endothelial')
else:
    for subtype in ['Arterial', 'Capillary', 'Capillary_high_MT', 'Venous']:
        names_celltype.append('_'.join(['Endothelial', subtype]))

names_celltype.append('Pericyte')

if data_name == 'Leng_2021_Nat_Neurosci':
    only_celltype = ['OPC', 'Endothelial', 'Pericyte']
else:
    only_celltype = ['OPC', 'Pericyte']

##### Output path #####
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-05-24-08-25-24-135401(pretrain_2epoch_lr1e-3_256dim)', data_name)
embedding_file = sc.read_h5ad(join(path_output, 'embedding.h5ad'))

filter_cells = ['Double_negative_Neuron', 'Double_positive_Neuron', 'GNLY_CD44_myeloid_sub1']
celltype_bools = np.invert([x in filter_cells for x in embedding_file.obs['celltype']])
embedding_file = embedding_file[celltype_bools].copy()

subtype_list = []
for celltype, subtype in zip(embedding_file.obs['celltype'], embedding_file.obs['subtype']):
    if celltype in only_celltype:
        subtype_list.append(celltype)
    else:
        subtype_list.append('_'.join([celltype, subtype]))

embedding_file.obs['type'] = subtype_list
embedding = embedding_file.X

path_fig = join(path_output, 'fig')
path_fig_label = join(path_output, 'fig_label')
makedirs(path_fig, exist_ok=True)
makedirs(path_fig_label, exist_ok=True)

embedding = StandardScaler().fit_transform(embedding)

n_neighbors = 15
min_dist = 0.1
umap_data = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='correlation', random_state=0).fit_transform(embedding)

groups_celltype = embedding_file.obs['type']
AC = [[5, 153, 107], [44, 129, 95], [96, 187, 102], [58, 190, 139], [73, 125, 75]]
MG = [[249, 126, 19], [250, 170, 0], [204, 123, 0], [255, 145, 60],
      [230, 117, 60], [244, 176, 40], [255, 184, 47], [224, 127, 0]]
OL = [[190, 59, 64], [190, 0, 30], [178, 5, 54], [242, 82, 49], 
      [177, 51, 27], [215, 0, 60], [237, 109, 90], [235, 34, 34]]
OPC = [[83, 55, 125]]
EXN = [[74, 169, 231], [42, 132, 185], [143, 181, 206], [28, 101, 127], [65, 199, 253],
       [94, 128, 153], [1, 171, 167], [151, 195, 196], [123, 170, 172], [1, 217, 213],
       [95, 197, 182], [159, 221, 211], [84, 125 ,122], [145, 230, 233], [71, 155, 159],
       [115, 237, 226], [0, 187, 192], [41, 124, 113], [11, 119, 165], [13, 93, 126]]
INN = [[137, 30, 168], [161, 19, 151], [200, 110, 255], [227, 84, 213], [134, 71, 210],
       [181, 121, 246], [144, 69, 185], [193, 0, 199], [200, 160, 255], [160, 0, 151],
       [115, 63, 176], [231, 163, 244], [142, 39, 173], [239, 143, 226], [146, 55, 187],
       [119, 78, 162], [211, 149, 255], [178, 73, 205], [180, 98, 233], [181, 137, 222],
       [184, 132, 237], [232, 125, 219], [228, 145, 255], [134, 47, 145], [180, 120, 245]]
if data_name == 'Leng_2021_Nat_Neurosci':
    EC = [[196, 236, 4]]
else:
    EC = [[183, 206, 0], [184, 235, 98], [147, 217, 43], [160, 228, 0]]
PC = [[240, 222, 0]]
colormap_celltype = np.array(AC + MG + OL + OPC + EXN + INN + EC + PC, dtype=np.float32) / 255.0
colormap_celltype = np.concatenate([colormap_celltype, np.ones((len(names_celltype), 1), dtype=np.float32)], axis=1)
fig_name_celltype = join(path_fig, f'umap_subtype.png')
fig_name_celltype_label = join(path_fig_label, f'umap_subtype.png')
plot_umap(umap_data, names_celltype, colormap_celltype, groups_celltype, fig_name_celltype, subtype_list)
plot_umap(umap_data, names_celltype, colormap_celltype, groups_celltype, fig_name_celltype_label, subtype_list, legend=True)
