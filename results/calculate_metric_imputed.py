import scanpy as sc
import numpy as np
import pandas as pd
from scipy import io as sio
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
# path_output = join('/home/gyutaek/hdd2/Results/MAGIC_UMI4000', data_name)
# path_output = join('/home/gyutaek/hdd2/Results/DCA_UMI4000', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scHyena_baseline/2024-09-27-07-35-14-850417(imputation_jung_UMI4000)', data_name)
# path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-09-25-00-12-28-112933(imputation_zhu_5epoch_lr1e-4_256dim_UMI4000)', data_name)

metadata = pd.read_csv(join('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype', data_name, 'metadata_imputation.csv'), index_col=0)
cell_names = metadata.index

label = sc.read_h5ad(join('/home/gyutaek/hdd2/Data/Process/MAGIC_UMI4000', data_name, 'data.h5ad'))
filtered_cells = [x in cell_names for x in label.obs.index]
label = label[filtered_cells].copy()
label = label.X

imputed = sc.read_h5ad(join(path_output, 'imputed_simul.h5ad'))
filtered_cells = [x in cell_names for x in imputed.obs.index]
imputed = imputed[filtered_cells].copy()
imputed.X[imputed.X < 0] = 0
imputed = imputed.X

mse = np.zeros(5)
corr = np.zeros(5)

for n in range(5):
    label_list = np.array([])
    output_list = np.array([])

    # imputed = sc.read_h5ad(join(path_output, f'imputed_simul_{n}.h5ad'))
    # imputed.X[imputed.X < 0] = 0
    # imputed = imputed.X

    for i in tqdm(range(label.shape[0])):
        label_cell = label[i]
        output_cell = imputed[i]
        nonzero_idx = np.nonzero(label_cell)[0]
        sub_idx = nonzero_idx[n::5]
        label_cell = label_cell[sub_idx]
        output_cell = output_cell[sub_idx]
        label_list = np.concatenate([label_list, label_cell], axis=0)
        output_list = np.concatenate([output_list, output_cell], axis=0)

    mse[n] = sum((label_list - output_list) ** 2) / len(label_list)
    corr[n] = np.corrcoef(label_list, output_list)[0, 1]

np.save(join(path_output, 'mse.npy'), mse)
np.save(join(path_output, 'corr.npy'), corr)

print('MSE: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(mse[0], mse[1], mse[2], mse[3], mse[4]))
print('Corr: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(corr[0], corr[1], corr[2], corr[3], corr[4]))
