import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from anndata import AnnData


### Code for simulating doublets
def doublet_simul(adata, doublet_ratio=0.1):
    # Filter out doublets and unidentified cells
    adata = adata[adata.obs['celltype'] != 'Doublet']
    adata = adata[adata.obs['celltype'] != 'Unidentified']
    n_genes = adata.shape[1]
    donors = adata.obs['donor'].unique()

    adata_doublet = None

    for donor in tqdm(donors):
        # Calculate the number of doublets to simulate
        adata_donor = adata[adata.obs['donor'] == donor].copy()
        n_cells = adata_donor.shape[0]
        n_doublets = int(n_cells * doublet_ratio)

        # Metadata for doublets
        doublets = np.zeros((n_doublets, n_genes), dtype=np.float32)
        obs_index = [donor + '_doublet_' + str(i) for i in range(n_doublets)]
        obs_donor = [donor for _ in range(n_doublets)]
        obs_celltype = ['Doublet' for _ in range(n_doublets)]
        obs_disease = [adata_donor.obs['Disease'][0] for i in range(n_doublets)]
        obs_subtype = ['Doublet' for i in range(n_doublets)]
        obs_subcluster = ['Doublet' for i in range(n_doublets)]

        obs = pd.DataFrame({'donor': obs_donor, 'celltype': obs_celltype, 'Disease': obs_disease, 'subtype': obs_subtype, 'subcluster': obs_subcluster}, index=obs_index)

        # Select two cells randomly and average their expression
        for n in range(n_doublets):
            idxs = np.random.choice(n_cells, 2, replace=False)
            doublets[n] = (adata_donor.X[idxs[0]] + adata_donor.X[idxs[1]]) / 2
        
        if adata_doublet is None:
            adata_doublet = AnnData(doublets, obs=obs, var=adata.var)
        else:
            adata_doublet = ad.concat([adata_doublet, AnnData(doublets, obs=obs, var=adata.var)], join='outer')
        del adata_donor
    
    return adata, adata_doublet
