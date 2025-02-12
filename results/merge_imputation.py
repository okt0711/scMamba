import scanpy as sc
import numpy as np
import anndata as ad
from os.path import join
from ipdb import set_trace as st


# data_name = 'Lau_2020_PNAS'
# test_donors = ['GSM4775564_AD5', 'GSM4775572_AD21', 'GSM4775573_NC3', 'GSM4775581_NC18']

# data_name = 'Leng_2021_Nat_Neurosci'
# test_donors = ['GSM4432641_SFG5', 'GSM4432646_EC1', 'GSM4432650_EC7', 'GSM4432653_EC8']

# data_name = 'Smajic_2022_brain'
# test_donors = ['C4', 'PD3']

data_name = 'Zhu_2022_bioRxiv'
test_donors = ['GSM6106342_HSDG10HC', 'GSM6106348_HSDG199PD']

# data_name = 'Jung_2022_unpublished'
# test_donors = ['AD_SN_X5738_X5732_X5720_X5704_X5665_X5626', 'DLB_FC_X5505_X5501_X5462_X5428_X5353_X5311',
#                'NO_SN_X5006_X4996_NO_FC_X5114_X5070_X5049', 'PD_SN_X5742_X5215_X5778',
#                'X5628NOHC', 'X5732ADHC']

# path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-09-25-00-12-28-112933(imputation_zhu_5epoch_lr1e-4_256dim_UMI4000)', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scHyena_baseline/2024-09-26-15-50-07-813488(imputation_zhu_UMI4000)', data_name)

print('Load imputed train data')
adata_train = sc.read_h5ad(join(path_output, 'imputed_train.h5ad'))

print('Load imputed test data')
adata_test = sc.read_h5ad(join(path_output, 'imputed.h5ad'))

adata = ad.concat([adata_train, adata_test], join='outer')

adata.X = adata.layers['raw'].astype('<f4')
del adata.layers['raw']
del adata.obs['scale_factor']

adata.write(join(path_output, 'imputed_merged.h5ad'))
