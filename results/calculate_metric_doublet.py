import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from os import listdir
from os.path import join
from tqdm import tqdm
from ipdb import set_trace as st


def cal_metrics(cm, labels, scores):
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    tnr = TN / (TN + FP)
    f1 = 2 * TP / ((TP + FP) + (TP + FN))
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    print("{:<15} {:<15.4f}".format("Precision", precision))
    print("{:<15} {:<15.4f}".format("Recall", recall))
    print("{:<15} {:<15.4f}".format("TNR", tnr))
    print("{:<15} {:<15.4f}".format("F1-score", f1))
    print("{:<15} {:<15.4f}".format("AUROC", auroc))
    print("{:<15} {:<15.4f}".format("AUPRC", auprc))


# data_name = 'Lau_2020_PNAS'
# data_name = 'Lau_2020_PNAS_simul'
# test_donors = ['GSM4775564_AD5', 'GSM4775572_AD21', 'GSM4775573_NC3', 'GSM4775581_NC18']

# data_name = 'Leng_2021_Nat_Neurosci'
# data_name = 'Leng_2021_Nat_Neurosci_simul'
# test_donors = ['GSM4432641_SFG5', 'GSM4432646_EC1', 'GSM4432650_EC7', 'GSM4432653_EC8']

# data_name = 'Smajic_2022_brain'
# data_name = 'Smajic_2022_brain_simul'
# test_donors = ['C4', 'PD3']

# data_name = 'Zhu_2022_bioRxiv'
data_name = 'Zhu_2022_bioRxiv_simul'
test_donors = ['GSM6106342_HSDG10HC', 'GSM6106348_HSDG199PD']

# data_name = 'Jung_2022_unpublished'
# test_donors = ['AD_SN_X5738_X5732_X5720_X5704_X5665_X5626','DLB_FC_X5505_X5501_X5462_X5428_X5353_X5311',
#                'NO_SN_X5006_X4996_NO_FC_X5114_X5070_X5049','PD_SN_X5742_X5215_X5778',
#                'X5628NOHC','X5732ADHC']

path_label = join('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-06-27-01-32-33-028977(doublet_zhu_simul_3cls_5epoch_lr1e-4_256dim)', data_name)

table = np.zeros((2, 2))

list_label = []
list_score = []

for i in tqdm(range(len(test_donors))):
    path_sub = join(path_output, test_donors[i])
    list_cell = sorted(listdir(path_sub))

    for j in range(len(list_cell)):
        file_label = torch.load(join(path_label, test_donors[i], list_cell[j]))
        celltype = file_label['celltype']
        if celltype == 'Doublet':
            label = 1
        else:
            label = 0
        list_label.append(label)
        file_output = torch.load(join(path_sub, list_cell[j]))
        celltype = file_output['prediction']
        if celltype == 'Doublet':
            output = 1
        else:
            output = 0
        table[label, output] += 1
        list_score.append(file_output['score'])

list_label = np.array(list_label)
list_score = np.array(list_score)

cal_metrics(table, list_label, list_score)