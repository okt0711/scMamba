import scanpy as sc
import numpy as np
import torch
import pandas as pd
from os import makedirs
from os.path import join
from tqdm import tqdm

from preprocessing import preprocess


### Code for processing scRNA-seq data and saving cells as individual .pt files
data_path = '/home/gyutaek/hdd2/Data/Raw/scRNA_seq'
data_name = 'Zhu_2022_bioRxiv'
path_save = join('/home/gyutaek/hdd2/Data/Process/scRNA_seq_process_subtype', data_name)
makedirs(path_save, exist_ok=True)

# Add metadata to the processed data
adata = sc.read_h5ad(join(data_path, data_name, 'matrix.h5ad'))

metadata = pd.read_csv(join(data_path, data_name, 'metadata.tsv'), sep='\t')
metadata_sub = pd.read_csv(join(data_path, 'metadata_subtype.tsv'), sep='\t')
metadata_sub = metadata_sub[metadata_sub['Study'] == data_name]
metadata_dis = pd.read_csv(join(data_path, data_name, 'disease.csv'))

adata.obs['celltype'] = 'Unidentified'
adata.obs['Disease'] = ''
adata.obs['subtype'] = 'Unidentified'
adata.obs['subcluster'] = 'Unidentified'

for donor, disease in zip(metadata_dis['donor'], metadata_dis['disease']):
    indices = adata.obs['donor'] == donor
    adata.obs.loc[indices, 'Disease'] = disease.upper()

for cell, celltype in zip(metadata['id'], metadata['celltype']):
    adata.obs.loc[cell, 'celltype'] = celltype

for cell, subtype, subcluster in zip(metadata_sub['id'], metadata_sub['subtype'], metadata_sub['subcluster']):
    adata.obs.loc[cell, 'subtype'] = subtype
    adata.obs.loc[cell, 'subcluster'] = subcluster

adata = preprocess(adata, batchkey='donor', size_factor=10000, select_hvg=False, gene_normalize=False, min_cells=False)

adata.write(join(path_save, 'matrix_processed.h5ad'))

# Save cells as individual .pt files
for cell in tqdm(adata.obs.index):
    cell_data = adata[cell]
    donor = cell_data.obs['donor'].tolist()[0]
    path_donor = join(path_save, donor)
    makedirs(path_donor, exist_ok=True)

    RNA = np.squeeze(cell_data.X.toarray())
    celltype = cell_data.obs['celltype'].tolist()[0]
    disease = cell_data.obs['Disease'].tolist()[0]
    subtype = cell_data.obs['subtype'].tolist()[0]
    subcluster = cell_data.obs['subcluster'].tolist()[0]

    torch.save({'RNA': RNA, 'celltype': celltype, 'disease': disease, 'subtype': subtype, 'subcluster': subcluster}, join(path_donor, cell + '.pt'))

# Save gene names as vocab.txt
list_gene = adata.var.index.tolist()

vocab_gene = open(join(path_save, 'vocab.txt'), 'w')
for gene in list_gene:
    gene = gene.strip().lower()
    vocab_gene.write(gene + '\n')
vocab_gene.close()
