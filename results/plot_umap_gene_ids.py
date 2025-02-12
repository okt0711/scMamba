import numpy as np
import torch
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from os import makedirs
from os.path import join
from ipdb import set_trace as st


def plot_umap(umap_data, names, colormap, groups, fig_name, gene_names):
    fig, ax = plt.subplots(1, figsize=(8, 7))

    for name in names:
        group = umap_data[groups == name, :]
        if name == 'NoneNoneNoneNoneNone':
            ax.scatter(group[:, 0], group[:, 1], linewidths=0, s=10, marker='o', label=name, color=[0.7, 0.7, 0.7, 1], alpha=0.5)
        else:
            ax.scatter(group[:, 0], group[:, 1], s=30, marker='o', label=name, color=colormap[names.index(name)], edgecolor='black', alpha=0.8)
            for i in range(len(group)):
                ax.text(group[i, 0]-0.28, group[i, 1]+0.11, gene_names[groups == name][i], fontsize=4, ha='center', va='center', weight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    # ax.set_xlim([min(umap_data[:, 0]) - 2, max(umap_data[:, 0]) + 2])
    # ax.set_ylim([min(umap_data[:, 1]) - 2, max(umap_data[:, 1]) + 2])
    # Line, Label = ax.get_legend_handles_labels()
    # fig.legend(Line, Label, loc='upper right')

    fig.savefig(fig_name, dpi=300)
    plt.close()

list_gene = open('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype/vocab.txt')
gene_ids = list_gene.readlines()
for i in range(len(gene_ids)):
    gene_ids[i] = gene_ids[i].strip()
list_gene.close()
gene_ids = gene_ids[1005:]

groups = np.array(['NoneNoneNoneNoneNone'] * len(gene_ids))
gene_names = np.array(['NoneNoneNoneNoneNone'] * len(gene_ids))
# genes_astrocyte = ['ensg00000131095', 'ensg00000079215', 'ensg00000110436', 'ensg00000160307', 'ensg00000171885', 'ensg00000144908']
# genes_microglia = ['ensg00000130203', 'ensg00000110651', 'ensg00000159189', 'ensg00000019582', 'ensg00000182578']
# genes_oligodendrocyte = ['ensg00000197971', 'ensg00000205927', 'ensg00000013297', 'ensg00000204655', 'ensg00000080822']
# genes_opc = ['ensg00000184221', 'ensg00000005513', 'ensg00000038427', 'ensg00000134853']
# genes_excitatory = ['ensg00000104888', 'ensg00000091664', 'ensg00000176884', 'ensg00000273079', 'ensg00000154146', 'ensg00000070808']
# genes_inhibitory = ['ensg00000157103', 'ensg00000204681', 'ensg00000128683', 'ensg00000136750', 'ensg00000183044']
# genes_endothelial = ['ensg00000102755', 'ensg00000117394', 'ensg00000184113']
# genes_pericyte = ['ensg00000113721', 'ensg00000076706', 'ensg00000166825', 'ensg00000107796']

genes_astrocyte = ['ensg00000131095', 'ensg00000171885', 'ensg00000160307', 'ensg00000144908', 'ensg00000135821',
                   'ensg00000079215', 'ensg00000134873', 'ensg00000080493', 'ensg00000152661', 'ensg00000041982']
genename_astrocyte = ['GFAP', 'AQP4', 'S100B', 'ALDH1L1', 'GLUL', 'SLC1A3', 'CLDN10', 'SLC4A4', 'GJA1', 'TNC']
genes_microglia = ['ensg00000169313', 'ensg00000168329', 'ensg00000182578', 'ensg00000095970', 'ensg00000169896',
                   'ensg00000183160', 'ensg00000105383', 'ensg00000173372', 'ensg00000103449', 'ensg00000184500']
genename_microglia = ['P2RY12', 'CX3CR1', 'CSF1R', 'TREM2', 'ITGAM', 'TMEM119', 'CD33', 'C1QA', 'SALL1', 'PROS1']
genes_oligodendrocyte = ['ensg00000204655', 'ensg00000197971', 'ensg00000205927', 'ensg00000123560', 'ensg00000173786',
                         'ensg00000100146', 'ensg00000105695', 'ensg00000065361', 'ensg00000013297', 'ensg00000144230']
genename_oligodendrocyte = ['MOG', 'MBP', 'OLIG2', 'PLP1', 'CNP', 'SOX10', 'MAG', 'ERBB3', 'CLDN11', 'GPR17']
genes_opc = ['ensg00000134853', 'ensg00000173546', 'ensg00000105894', 'ensg00000072163', 'ensg00000041982',
             'ensg00000164434', 'ensg00000124766', 'ensg00000132692', 'ensg00000090932', 'ensg00000109846']
genename_opc = ['PDGFRA', 'CSPG4', 'PTN', 'LIMS2', 'TNC', 'FABP7', 'SOX4', 'BCAN', 'DLL3', 'CRYAB']
genes_neuron = ['ensg00000078018', 'ensg00000167281', 'ensg00000008056', 'ensg00000176884', 'ensg00000132639',
                'ensg00000070808', 'ensg00000077279', 'ensg00000258947', 'ensg00000104435', 'ensg00000164600']
genename_neuron = ['MAP2', 'RBFOX3', 'SYN1', 'GRIN1', 'SNAP25', 'CAMK2A', 'DCX', 'TUBB3', 'STMN2', 'NEUROD6']
genes_endothelial = ['ensg00000261371', 'ensg00000179776', 'ensg00000112715', 'ensg00000184113', 'ensg00000102755',
                     'ensg00000128052', 'ensg00000110799', 'ensg00000149564', 'ensg00000130300', 'ensg00000106991']
genes_pericyte = ['ensg00000113721', 'ensg00000143248', 'ensg00000107796', 'ensg00000173546', 'ensg00000069431',
                  'ensg00000121361', 'ensg00000106484', 'ensg00000149591', 'ensg00000076706']

for gene in genes_astrocyte:
    groups[gene_ids.index(gene)] = 'Astrocyte'
    gene_names[gene_ids.index(gene)] = genename_astrocyte[genes_astrocyte.index(gene)]
for gene in genes_microglia:
    groups[gene_ids.index(gene)] = 'Microglia'
    gene_names[gene_ids.index(gene)] = genename_microglia[genes_microglia.index(gene)]
for gene in genes_oligodendrocyte:
    groups[gene_ids.index(gene)] = 'Oligodendrocyte'
    gene_names[gene_ids.index(gene)] = genename_oligodendrocyte[genes_oligodendrocyte.index(gene)]
# for gene in genes_opc:
#     groups[gene_ids.index(gene)] = 'OPC'
#     gene_names[gene_ids.index(gene)] = genename_opc[genes_opc.index(gene)]
# for gene in genes_excitatory:
#     groups[gene_ids.index(gene)] = 'Excitatory'
# for gene in genes_inhibitory:
#     groups[gene_ids.index(gene)] = 'Inhibitory'
for gene in genes_neuron:
    groups[gene_ids.index(gene)] = 'Neuron'
    gene_names[gene_ids.index(gene)] = genename_neuron[genes_neuron.index(gene)]
# for gene in genes_endothelial:
#     groups[gene_ids.index(gene)] = 'Endothelial'
# for gene in genes_pericyte:
#     groups[gene_ids.index(gene)] = 'Pericyte'

# genes_neurogenesis = ['ensg00000181449', 'ensg00000132688', 'ensg00000077279', 'ensg00000136535', 'ensg00000162992',
#                       'ensg00000139352', 'ensg00000148400', 'ensg00000007372', 'ensg00000170370']
# genes_synaptic = ['ensg00000176884', 'ensg00000155511', 'ensg00000145864', 'ensg00000067715', 'ensg00000008056',
#                   'ensg00000251322', 'ensg00000132535', 'ensg00000157103', 'ensg00000220205']
# genes_neuroinflammation = ['ensg00000162711', 'ensg00000095970', 'ensg00000129226', 'ensg00000204287', 'ensg00000182578',
#                            'ensg00000173369', 'ensg00000204472']

# for gene in genes_neurogenesis:
#     groups[gene_ids.index(gene)] = 'Neurogenesis'
# for gene in genes_synaptic:
#     groups[gene_ids.index(gene)] = 'Synaptic'
# for gene in genes_neuroinflammation:
#     groups[gene_ids.index(gene)] = 'Neuroinflammation'


# gene_id_embedding = np.load(join(path_output,'gene_id_embedding.npy'))
path_ckpt = '/home/gyutaek/ssd1/Projects/scMamba/outputs/2024-05-24/08-25-24-135401/checkpoints/train/loss.ckpt'
state_dict = torch.load(path_ckpt, map_location='cpu')
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
    state_dict["state_dict"], "model."
)
model_state_dict = state_dict["state_dict"]
gene_id_embedding = model_state_dict['backbone.gene_id_embedding.weight'][1005:-1].numpy()

path_output = '/home/gyutaek/hdd2/Results/scMamba/2024-05-24-08-25-24-135401(pretrain_2epoch_lr1e-3_256dim)'
path_fig = join(path_output, 'fig')
makedirs(path_fig, exist_ok=True)

# gene_id_embedding = StandardScaler().fit_transform(gene_id_embedding)

n_neighbors = 15
min_dist = 0.1
umap_data = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', random_state=0).fit_transform(gene_id_embedding)
# umap_data = PCA(n_components=2).fit_transform(gene_id_embedding)
# umap_data = TSNE(n_components=2, perplexity=10, random_state=0).fit_transform(gene_id_embedding)

# names_celltype = ['NoneNoneNoneNoneNone', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Excitatory', 'Inhibitory', 'Endothelial', 'Pericyte']
# colormap = np.array([[62, 158, 100], [255, 153, 0], [217, 49, 55], [83, 55, 125], [45, 140, 184], [184, 100, 204], [196, 236, 4], [240, 222, 0]], dtype=np.float32) / 255.0
# names_celltype = ['NoneNoneNoneNoneNone', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'Neuron', 'Endothelial', 'Pericyte']
# colormap = np.array([[62, 158, 100], [255, 153, 0], [217, 49, 55], [45, 140, 184], [196, 236, 4], [240, 222, 0]], dtype=np.float32) / 255.0
names_celltype = ['NoneNoneNoneNoneNone', 'Astrocyte', 'Microglia', 'Oligodendrocyte', 'Neuron']
colormap = np.array([[62, 158, 100], [255, 153, 0], [217, 49, 55], [45, 140, 184]], dtype=np.float32) / 255.0
# names_celltype = ['NoneNoneNoneNoneNone', 'Neurogenesis', 'Synaptic', 'Neuroinflammation']
# colormap = np.array([[62, 158, 100], [217, 49, 55], [45, 140, 184]], dtype=np.float32) / 255.0
colormap = np.concatenate([colormap, np.ones((len(names_celltype) - 1, 1), dtype=np.float32)], axis=1)
colormap = np.concatenate((np.array([[0.5, 0.5, 0.5, 1]]), colormap))
fig_name = join(path_fig, 'umap_genes.png')
# fig_name = join(path_fig, 'pca_genes.png')
plot_umap(umap_data, names_celltype, colormap, groups, fig_name, gene_names)