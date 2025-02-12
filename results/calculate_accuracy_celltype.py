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

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * TP / ((TP + FP) + (TP + FN))

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    weighted_precision = np.sum(weight * precision)
    weighted_recall = np.sum(weight * recall)
    weighted_f1 = np.sum(weight * f1)
    micro_f1 = np.trace(cm) / np.sum(cm)

    print("{:<15} {:<15} {:<15} {:<15}".format("Class", "Precision", "Recall", "F1-score"))
    for i in range(len(cls)):
        print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format(cls[i], precision[i], recall[i], f1[i]))
    print("{:<15} {:<15} {:<15} {:<15.4f}".format("Micro", "", "", micro_f1))
    print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format("Macro", macro_precision, macro_recall, macro_f1))
    print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format("Weighted", weighted_precision, weighted_recall, weighted_f1))


# data_name = 'Lau_2020_PNAS'
# test_donors = ['GSM4775564_AD5', 'GSM4775572_AD21', 'GSM4775573_NC3', 'GSM4775581_NC18']

# data_name = 'Leng_2021_Nat_Neurosci'
# test_donors = ['GSM4432641_SFG5', 'GSM4432646_EC1', 'GSM4432650_EC7', 'GSM4432653_EC8']

# data_name = 'Smajic_2022_brain'
# test_donors = ['C4', 'PD3']

# data_name = 'Zhu_2022_bioRxiv'
# test_donors = ['GSM6106342_HSDG10HC', 'GSM6106348_HSDG199PD']

data_name = 'Jung_2022_unpublished_total'
test_donors = ['AD_SN_X5738_X5732_X5720_X5704_X5665_X5626','DLB_FC_X5505_X5501_X5462_X5428_X5353_X5311',
               'NO_SN_X5006_X4996_NO_FC_X5114_X5070_X5049','PD_SN_X5742_X5215_X5778',
               'X5628NOHC','X5732ADHC']

path_label = join('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-06-02-14-08-35-164237(celltype_jung_middle_10epoch_lr1e-4_256dim_3cls)', data_name)

celltype_list = ['Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Excitatory', 'Inhibitory', 'Endothelial', 'Pericyte']

table = np.zeros((len(celltype_list), len(celltype_list)))

for i in tqdm(range(len(test_donors))):
    path_sub = join(path_output, test_donors[i])
    list_cell = sorted(listdir(path_sub))

    for j in range(len(list_cell)):
        file_label = torch.load(join(path_label, test_donors[i], list_cell[j]))
        label = celltype_list.index(file_label['celltype'])
        file_output = torch.load(join(path_sub, list_cell[j]))
        output = celltype_list.index(file_output['prediction'])
        table[label, output] += 1

cal_f1_score(table, celltype_list)