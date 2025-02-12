import torch
import numpy as np
from os import listdir
from os.path import join
from tqdm import tqdm
from ipdb import set_trace as st


def cal_f1_score(cm, cls):
    TP = np.array([cm[i, i] for i in range(cm.shape[0])])
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    weight = np.sum(cm, axis=1) / np.sum(cm)
    is_nonempty = weight != 0

    precision = TP / (TP + FP)
    precision[~is_nonempty] = -100
    precision[np.isnan(precision)] = 0
    recall = TP / (TP + FN)
    recall[~is_nonempty] = -100
    recall[np.isnan(recall)] = 0
    f1 = 2 * TP / ((TP + FP) + (TP + FN))
    f1[~is_nonempty] = -100
    f1[np.isnan(f1)] = 0

    macro_precision = np.mean(precision[is_nonempty])
    macro_recall = np.mean(recall[is_nonempty])
    macro_f1 = np.mean(f1[is_nonempty])
    weighted_precision = np.sum(weight * precision)
    weighted_recall = np.sum(weight * recall)
    weighted_f1 = np.sum(weight * f1)
    micro_f1 = np.trace(cm) / np.sum(cm)

    print("{:<50} {:<15} {:<15} {:<15}".format("Class", "Precision", "Recall", "F1-score"))
    for i in range(len(cls)):
        print("{:<50} {:<15.4f} {:<15.4f} {:<15.4f}".format(cls[i], precision[i], recall[i], f1[i]))
    print("{:<50} {:<15} {:<15} {:<15.4f}".format("Micro", "", "", micro_f1))
    print("{:<50} {:<15.4f} {:<15.4f} {:<15.4f}".format("Macro", macro_precision, macro_recall, macro_f1))
    print("{:<50} {:<15.4f} {:<15.4f} {:<15.4f}".format("Weighted", weighted_precision, weighted_recall, weighted_f1))


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

for subtype in ['APOE_astrocyte_11', 'APOE_astrocyte_16', 'APOE_astrocyte_7', 'CHII3L1_astrocyte_10',
                'DPP10_astrocyte_12', 'DPP10_astrocyte_13', 'DPP10_astrocyte_4', 'DPP10_astrocyte_5', 'DPP10_astrocyte_8',
                'GRM3_astrocyte_0', 'GRM3_astrocyte_1', 'GRM3_astrocyte_2', 'GRM3_astrocyte_3', 'Intermediate_6', 'Intermediate_9']:
    celltype_list.append('_'.join(['Astrocyte', subtype]))

for subtype in ['Homeostatic_0', 'Homeostatic_3', 'Homeostatic_5', 'Homeostatic_defficent_1', 'InflammatoryII_7', 'InflammatoryI_9',
                'Lipid_processing_4', 'Phago_Infla_intermediate_11', 'Phagocytic_6', 'Phagocytic_8', 'Ribosomal_genesis_10', 'Ribosomal_genesis_2']:
    celltype_list.append('_'.join(['Microglia', subtype]))

for subtype in ['CAMK2D_Oligo_13', 'OPALIN_FRY_Oligo_8', 'OPALIN_Oligo_1', 'OPALIN_Oligo_10', 'OPALIN_Oligo_4', 'OPALIN_Oligo_7',
                'OPALIN_high_Oligo_3', 'RBFOX1_Oligo_5', 'RBFOX1_Oligo_6', 'RBFOX1_high_Oligo_12', 'RBFOX1_high_Oligo_2',
                'highMT_11', 'intermediate_0', 'intermediate_9']:
    celltype_list.append('_'.join(['Oligodendrocyte', subtype]))

celltype_list.append('OPC')

for subtype in ['CALB_30', 'L2-4_Lamp5_1', 'L2-4_Lamp5_20', 'L2-4_Lamp5_3', 'L2-4_Lamp5_33', 'L2-4_Lamp5_37', 'L2-4_SYT2_21', 'L2_3_11', 'L2_3_2', 'L2_3_8', 'L2_3_9',
                'L4/5_RORB_GABRG1_24', 'L4/5_RORB_GABRG1_4', 'L4/5_RORB_LINC02196_23', 'L4/5_RORB_PCP4_15', 'L4/5_RORB_PCP4_5', 'L4/5_RORB_PLCH1_MME_13', 'L4/5_RORB_RPRM_10',
                'L4_RORB_COL5A2_PLCH1_26', 'L4_RORB_COL5A2_PLCH1_6', 'L5/6_NFIA_THEMIS_28', 'L5/6_NFIA_THEMIS_7', 'L5/6_NXPH2_19', 'L5_ET_SYT2_31', 'L6_HPSE2_NR4A2_NTNG2_22',
                'L6b_PCSK5_SEMA3D_18', 'L6b_PCSK5_SEMA3D_29', 'L6b_PCSK5_SULF1_14', 'NRGN_12', 'NRGN_16', 'NRGN_27', 'RELN_CHD7_32', 'SOX6_17', 'high_MT_0']:
    celltype_list.append('_'.join(['Excitatory', subtype]))

for subtype in ['ALCAM_TRPM3_17', 'CUX2_MSR1_16', 'ENOX2_SPHKAP_25', 'FBN2_EPB41L4A_19', 'GPC5_RIT2_12', 'LAMP5_CA13_11', 'LAMP5_CA13_34', 'LAMP5_NRG1_1', 'LAMP5_NRG1_28',
                'LAMP5_RELN_20', 'PAX6_CA4_21', 'PTPRK_FAM19A1_26', 'PVALB_CA8_5', 'PVALB_SULF1_HTR4_0', 'PVALB_SULF1_HTR4_13', 'RYR3_TSHZ2_29', 'RYR3_TSHZ2_9', 'RYR3_TSHZ2_VIP_THSD7B_27',
                'SGCD_PDE3A_23', 'SN1_2', 'SN2_10', 'SORCS1_TTN_24', 'SST_MAFB_8', 'SST_NPY_39', 'VIP_ABI3BP_THSD7B_22', 'VIP_ABI3BP_THSD7B_4', 'VIP_CLSTN2_6', 'VIP_THSD7B_36', 'VIP_TSHZ2_14',
                'high_MT_15', 'high_MT_3', 'high_MT_32', 'high_MT_35', 'high_MT_37', 'high_MT_44', 'high_MT_7']:
    celltype_list.append('_'.join(['Inhibitory', subtype]))

if data_name == 'Leng_2021_Nat_Neurosci':
    celltype_list.append('Endothelial')
    only_celltype = ['OPC', 'Endothelial', 'Pericyte']
else:
    for subtype in ['Arterial_3', 'Arterial_4', 'Capillary_12', 'Capillary_2', 'Capillary_6', 'Capillary_8', 'Capillary_9',
                    'Capillary_high_MT_0', 'Capillary_high_MT_11', 'Capillary_high_MT_13', 'Capillary_high_MT_5', 'Venous_1', 'Venous_10', 'Venous_7']:
        celltype_list.append('_'.join(['Endothelial', subtype]))
    only_celltype = ['OPC', 'Pericyte']

celltype_list.append('Pericyte')

path_label = join('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-10-08-00-17-58-958958(subcluster_jung_3cls_10epoch_lr1e-4_256dim)', data_name)

table = np.zeros((len(celltype_list), len(celltype_list)))

for i in tqdm(range(len(test_donors))):
    path_sub = join(path_output, test_donors[i])
    list_cell = sorted(listdir(path_sub))

    for j in range(len(list_cell)):
        file_label = torch.load(join(path_label, test_donors[i], list_cell[j]))
        if file_label['celltype'] in only_celltype:
            label = celltype_list.index(file_label['celltype'])
        else:
            label = celltype_list.index('_'.join([file_label['celltype'], file_label['subcluster']]))
        file_output = torch.load(join(path_sub, list_cell[j]))
        output = celltype_list.index(file_output['prediction'])
        table[label, output] += 1

cal_f1_score(table, celltype_list)
