import scanpy as sc
import numpy as np
import torch
import anndata as ad
from os import makedirs
from os.path import join
from ipdb import set_trace as st


# data_name = 'Lau_2020_PNAS'
# data_name = 'Leng_2021_Nat_Neurosci'
# data_name = 'Smajic_2022_brain'
data_name = 'Zhu_2022_bioRxiv'
# data_name = 'Jung_2022_unpublished'

path_imputed = join('/home/gyutaek/hdd2/Results/scHyena_baseline/2024-09-26-15-50-07-813488(imputation_zhu_UMI4000)', data_name, 'imputed_train.h5ad')
path_output = join('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype', data_name + '_imputed_schyena')
makedirs(path_output, exist_ok=True)

print('Load imputed data')
adata = sc.read_h5ad(path_imputed)
adata.X[adata.X < 0] = 0.0

for i in range(adata.shape[0]):
    donor = adata.obs['donor'][i]
    path_donor = join(path_output, donor)
    makedirs(path_donor, exist_ok=True)

    RNA = adata.X[i]
    celltype = adata.obs['celltype'][i]
    disease = adata.obs['Disease'][i]
    subtype = adata.obs['subtype'][i]
    subcluster = adata.obs['subcluster'][i]
    cell = adata.obs.index[i]

    torch.save({'RNA': RNA, 'celltype': celltype, 'disease': disease, 'subtype': subtype, 'subcluster': subcluster}, join(path_donor, cell + '.pt'))

