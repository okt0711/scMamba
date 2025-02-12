import scanpy as sc
import torch
import pandas as pd
from os.path import join
from tqdm import tqdm


### Code for adding metadata to the processed data and save cells as individual .pt files
data_path = '/home/gyutaek/hdd2/Data/Raw/scRNA_seq'
data_name = 'Zhu_2022_bioRxiv'
path_save = join('/home/gyutaek/hdd2/Data/Process/scRNA_seq_process_subtype', data_name)

# Add metadata to the processed data and save new h5ad file
metadata_sub = pd.read_csv(join(data_path, 'metadata_subtype.tsv'), sep='\t')
metadata_sub = metadata_sub[metadata_sub['Study'] == data_name]
metadata_sub = metadata_sub[metadata_sub['celltype'] == 'Excitatory']
cells = metadata_sub['id'].tolist()

adata = sc.read_h5ad(join(path_save, 'matrix_processed.h5ad'))
adata.obs['subtype'] = adata.obs['subtype'].astype(str)
adata.obs['subcluster'] = adata.obs['subcluster'].astype(str)
for cell in tqdm(adata.obs.index):
    if cell not in cells:
        continue
    subtype = metadata_sub[metadata_sub['id'] == cell]['subtype'].tolist()[0]
    subcluster = metadata_sub[metadata_sub['id'] == cell]['subcluster'].tolist()[0]
    adata.obs.loc[cell, 'subtype'] = subtype
    adata.obs.loc[cell, 'subcluster'] = subcluster

adata.write(join(path_save, 'matrix_processed.h5ad'))

# Save cells as individual .pt files
for cell in tqdm(adata.obs.index):
    cell_data = adata[cell]
    donor = cell_data.obs['donor'].tolist()[0]
    path_donor = join(path_save, donor)

    data = torch.load(join(path_donor, cell + '.pt'))
    data['subtype'] = cell_data.obs['subtype'].tolist()[0]
    data['subcluster'] = cell_data.obs['subcluster'].tolist()[0]
    torch.save(data, join(path_donor, cell + '.pt'))

del adata
