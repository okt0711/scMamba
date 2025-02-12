from tqdm import tqdm
from time import sleep
from numpy import unique, asarray
from numpy.random import choice, seed
from scipy.sparse import issparse

import scanpy as sc


### Code for preprocessing data
### This code is based on the preprocessing code from sciPENN (https://github.com/jlakkis/sciPENN)
def preprocess(adata, batchkey=None, gene_list=[], select_hvg=True, cell_normalize=True, size_factor=None,
               log_normalize=True, gene_normalize=True, min_cells=30, min_genes=200, n_top_genes=1000, hvgs=None):

    if batchkey is not None:
        adata.obs['batch'] = ['DS' + ' ' + x for x in adata.obs[batchkey]]
    else:
        adata.obs['batch'] = 'DS'

    gene_set = set(gene_list)

    if min_genes:
        print("\nQC Filtering Cells")

        cell_filter = (adata.X > 10 ** (-8)).sum(axis=1) >= min_genes
        adata = adata[cell_filter].copy()

    if min_cells:
        print("\nQC Filtering Genes")

        bools = (adata.X > 10 ** (-8)).sum(axis=0) > min_cells
        genes = adata.var.index[asarray(bools)[0]]
        genes = asarray(genes).reshape((-1,))
        features = set(genes)
        features.update(gene_set)
        features = list(features)
        features.sort()

        adata = adata[:, features].copy()

    adata.layers["raw"] = adata.X.copy()

    if cell_normalize:
        print("\nNormalizing Cells")

        sc.pp.normalize_total(adata, target_sum=size_factor, key_added="scale_factor")

    if log_normalize:
        print("\nLog-Normalizing Data")

        sc.pp.log1p(adata)

    if select_hvg:
        if hvgs is None:
            print("\nFinding HVGs")

            tmp = adata.copy()

            if not cell_normalize or not log_normalize:
                print("Warning, highly variable gene selection may not be accurate if expression is not cell normalized and log normalized")

            if len(tmp) > 10 ** 5:
                seed(123)
                idx = choice(range(len(tmp)), 10 ** 5, False)
                tmp = tmp[idx].copy()

            sc.pp.highly_variable_genes(tmp, min_mean=0.0125, max_mean=3, min_disp=0.5,
                                        n_bins=20, subset=False, batch_key='batch', n_top_genes=n_top_genes)
            hvgs = tmp.var.index[tmp.var['highly_variable']].copy()
            tmp = None

        gene_set.update(set(hvgs))
        gene_set = list(gene_set)
        gene_set.sort()
        adata = adata[:, gene_set].copy()

    make_dense(adata)

    if gene_normalize:
        patients = unique(adata.obs['batch'].values)

        print("\nNormalizing Gene Data by Batch")
        sleep(1)

        for patient in tqdm(patients):
            indices = [x == patient for x in adata.obs['batch']]
            sub_adata = adata[indices].copy()
            sc.pp.scale(sub_adata)

            adata[indices] = sub_adata.X.copy()
            adata.var[patient + ' mean'] = sub_adata.var['mean'].values
            adata.var[patient + ' std'] = sub_adata.var['std'].values

    return adata


def make_dense(anndata):
    if issparse(anndata.X):
        tmp = anndata.X.copy()
        anndata.X = tmp.copy().toarray()
