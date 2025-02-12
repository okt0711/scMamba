import scanpy as sc
import pandas as pd
from os.path import join


### Code for saving metadata as a .csv file
data_path = '/home/gyutaek/hdd2/Data/Process/scRNA_seq_process_subtype'
data_name = 'Jung_2022_unpublished'

adata = sc.read_h5ad(join(data_path, data_name, 'matrix_processed.h5ad'))

### For imputation task, filter cells with scale_factor >= 4000.0
# filter_cells = adata.obs['scale_factor'] >= 4000.0
# adata = adata[filter_cells].copy()

df = pd.DataFrame(index=adata.obs.index)
df['celltype'] = adata.obs['celltype']
df['Disease'] = adata.obs['Disease']
df['subtype'] = adata.obs['subtype']
df['subcluster'] = adata.obs['subcluster']
df['donor'] = adata.obs['donor']

df.to_csv(join(data_path, data_name, 'metadata.csv'), index=True)
### For imputation task
# df.to_csv(join(data_path, data_name, 'metadata_imputation.csv'), index=True)
