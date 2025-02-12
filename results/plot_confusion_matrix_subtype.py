import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from os import makedirs, listdir
from os.path import join
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from ipdb import set_trace as st


# data_name = 'Lau_2020_PNAS'
# test_donors = ['GSM4775564_AD5', 'GSM4775572_AD21', 'GSM4775573_NC3', 'GSM4775581_NC18']

# data_name = 'Leng_2021_Nat_Neurosci'
# test_donors = ['GSM4432641_SFG5', 'GSM4432646_EC1', 'GSM4432650_EC7', 'GSM4432653_EC8']

# data_name = 'Smajic_2022_brain'
# test_donors = ['C4', 'PD3']

# data_name = 'Zhu_2022_bioRxiv'
# test_donors = ['GSM6106342_HSDG10HC', 'GSM6106348_HSDG199PD']

data_name = 'Jung_2022_unpublished'
test_donors = ['AD_SN_X5738_X5732_X5720_X5704_X5665_X5626','DLB_FC_X5505_X5501_X5462_X5428_X5353_X5311',
               'NO_SN_X5006_X4996_NO_FC_X5114_X5070_X5049','PD_SN_X5742_X5215_X5778',
               'X5628NOHC','X5732ADHC']

celltype_list = []

for subtype in ['APOE_astrocyte', 'CHII3L1_astrocyte', 'DPP10_astrocyte', 'GRM3_astrocyte', 'Intermediate']:
    celltype_list.append('_'.join(['Astrocyte', subtype]))

for subtype in ['Homeostatic', 'Homeostatic_defficent', 'InflammatoryI', 'InflammatoryII', 'Lipid_processing', 'Phago_Infla_intermediate', 'Phagocytic', 'Ribosomal_genesis']:
    celltype_list.append('_'.join(['Microglia', subtype]))

for subtype in ['CAMK2D_Oligo', 'OPALIN_FRY_Oligo', 'OPALIN_Oligo', 'OPALIN_high_Oligo', 'RBFOX1_Oligo', 'RBFOX1_high_Oligo', 'highMT', 'intermediate']:
    celltype_list.append('_'.join(['Oligodendrocyte', subtype]))

celltype_list.append('OPC')

for subtype in ['CALB', 'L2-4_Lamp5', 'L2-4_SYT2', 'L2_3', 'L4/5_RORB_GABRG1', 'L4/5_RORB_LINC02196', 'L4/5_RORB_PCP4', 'L4/5_RORB_PLCH1_MME', 'L4/5_RORB_RPRM', 'L4_RORB_COL5A2_PLCH1',
                'L5/6_NFIA_THEMIS', 'L5/6_NXPH2', 'L5_ET_SYT2', 'L6_HPSE2_NR4A2_NTNG2', 'L6b_PCSK5_SEMA3D', 'L6b_PCSK5_SULF1', 'NRGN', 'RELN_CHD7', 'SOX6', 'high_MT']:
    celltype_list.append('_'.join(['Excitatory', subtype]))

for subtype in ['ALCAM_TRPM3', 'CUX2_MSR1', 'ENOX2_SPHKAP', 'FBN2_EPB41L4A', 'GPC5_RIT2', 'LAMP5_CA13', 'LAMP5_NRG1', 'LAMP5_RELN', 'PAX6_CA4',
                'PTPRK_FAM19A1', 'PVALB_CA8', 'PVALB_SULF1_HTR4', 'RYR3_TSHZ2', 'RYR3_TSHZ2_VIP_THSD7B', 'SGCD_PDE3A', 'SN1', 'SN2',
                'SORCS1_TTN', 'SST_MAFB', 'SST_NPY', 'VIP_ABI3BP_THSD7B', 'VIP_CLSTN2', 'VIP_THSD7B', 'VIP_TSHZ2', 'high_MT']:
    celltype_list.append('_'.join(['Inhibitory', subtype]))

if data_name == 'Leng_2021_Nat_Neurosci':
    celltype_list.append('Endothelial')
    only_celltype = ['OPC', 'Endothelial', 'Pericyte']
else:
    for subtype in ['Arterial', 'Capillary', 'Capillary_high_MT', 'Venous']:
        celltype_list.append('_'.join(['Endothelial', subtype]))
    only_celltype = ['OPC', 'Pericyte']

celltype_list.append('Pericyte')

path_label = join('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-09-09-12-40-22-700200(subtype_jung_3cls_10epoch_lr1e-4_256dim_resample)', data_name)

path_fig = join(path_output, 'fig')
makedirs(path_fig, exist_ok=True)

table = np.zeros((len(celltype_list), len(celltype_list)))
truth = []
pred = []

for i in tqdm(range(len(test_donors))):
    path_sub = join(path_output, test_donors[i])
    list_cell = sorted(listdir(path_sub))

    for j in range(len(list_cell)):
        file_label = torch.load(join(path_label, test_donors[i], list_cell[j]))
        if file_label['celltype'] in only_celltype:
            label = celltype_list.index(file_label['celltype'])
        else:
            label = celltype_list.index('_'.join([file_label['celltype'], file_label['subtype']]))
        file_output = torch.load(join(path_sub, list_cell[j]))
        output = celltype_list.index(file_output['prediction'])
        truth.append(celltype_list[label])
        pred.append(celltype_list[output])
        table[label, output] += 1

sns.set(font_scale=2)

cm = confusion_matrix(truth, pred, labels=celltype_list)
div = cm.sum(axis=1, keepdims=True)
div[div == 0] = 1
cm = cm / div

fig, ax = plt.subplots(1, 1, figsize=(50, 50))
sns.heatmap(cm, ax=ax, xticklabels=celltype_list, yticklabels=celltype_list, cmap='viridis', vmin=0, vmax=1)
fig.savefig(path_fig + '/confusion_matrix.png', bbox_inches='tight', dpi=300)
