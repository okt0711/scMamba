import numpy as np
import scanpy as sc
import pickle
import networkx as nx
import scib
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.neighbors import kneighbors_graph
from os.path import join
from tqdm import tqdm
from ipdb import set_trace as st

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

# sc.pp.pca(imputed)
imputed.obsm['X'] = imputed.X

results = scib.metrics.metrics(
    imputed,
    adata_int=imputed,
    batch_key='donor',
    label_key='celltype',
    embed='X',
    isolated_labels_asw_=False,
    silhouette_=True,
    hvg_score_=False,
    graph_conn_=True,
    pcr_=True,
    isolated_labels_f1_=False,
    trajectory_=False,
    nmi_=True,
    ari_=True,
    cell_cycle_=False,
    kBET_=False,
    ilisi_=False,
    clisi_=False,
)

results = results[0].to_dict()

NMI_cell = results["NMI_cluster/label"]
ARI_cell = results['ARI_cluster/label']
ASW_cell = results['ASW_label']
AvgBIO = (NMI_cell + ARI_cell + ASW_cell) / 3

ASW_batch = results['ASW_label/batch']
graph_conn = results['graph_conn']
AvgBATCH = (ASW_batch + graph_conn) / 2

Overall = 0.6 * AvgBIO + 0.4 * AvgBATCH

print('NMI_cell: {:.3f}, ARI_cell: {:.3f}, ASW_cell: {:.3f}, AvgBIO: {:.3f}'.format(NMI_cell, ARI_cell, ASW_cell, AvgBIO))
print('ASW_batch: {:.3f}, Graph connectivity: {:.3f}, AvgBATCH: {:.3f}'.format(ASW_batch, graph_conn, AvgBATCH))
print('Overall: {:.3f}'.format(Overall))

with open(join(path_output, 'metric_v2.pickle'), 'wb') as f:
    pickle.dump({'NMI': NMI_cell, 'ARI': ARI_cell, 'ASW': ASW_cell, 'AvgBIO': AvgBIO,
                 'ASW_batch': ASW_batch, 'Graph connectivity': graph_conn, 'AvgBATCH': AvgBATCH,
                 'Overall': Overall}, f)
