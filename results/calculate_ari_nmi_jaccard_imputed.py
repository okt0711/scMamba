import numpy as np
import scanpy as sc
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_samples
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
# path_output = join('/home/gyutaek/hdd2/Results/scHyena_baseline/2024-09-27-07-35-14-850417(imputation_jung_UMI4000)', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-09-25-08-08-52-533110(imputation_jung_5epoch_lr1e-4_256dim_UMI4000)', data_name)
imputed = sc.read_h5ad(join(path_output, 'imputed.h5ad'))
imputed.X[imputed.X < 0] = 0

imputed = imputed[imputed.obs['celltype'] != 'Double_negative_Neuron']
imputed = imputed[imputed.obs['celltype'] != 'Double_positive_Neuron']
imputed = imputed[imputed.obs['celltype'] != 'Doublet']
imputed = imputed[imputed.obs['celltype'] != 'GNLY_CD44_myeloid_sub1']
imputed = imputed[imputed.obs['celltype'] != 'Unidentified']

sc.pp.neighbors(imputed, use_rep='X')

best_nmi = -1
best_resolution = None

for resolution in np.arange(0.1, 2.1, 0.1):
    sc.tl.louvain(imputed, resolution=resolution, key_added=f'louvain_res_{resolution}')
    nmi = normalized_mutual_info_score(imputed.obs['celltype'], imputed.obs[f'louvain_res_{resolution}'])
    if nmi > best_nmi:
        best_nmi = nmi
        best_resolution = resolution

ari = adjusted_rand_score(imputed.obs['celltype'], imputed.obs[f'louvain_res_{best_resolution}'])

asw = silhouette_samples(imputed.X, imputed.obs['celltype'])

print(f'NMI: {best_nmi}, ARI: {ari}, ASW: {asw.mean()}, ASW(norm): {(asw.mean() + 1) / 2}')
