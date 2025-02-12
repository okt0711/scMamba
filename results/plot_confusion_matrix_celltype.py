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

data_name = 'Jung_2022_unpublished_total'
test_donors = ['AD_SN_X5738_X5732_X5720_X5704_X5665_X5626', 'DLB_FC_X5505_X5501_X5462_X5428_X5353_X5311',
               'NO_SN_X5006_X4996_NO_FC_X5114_X5070_X5049', 'PD_SN_X5742_X5215_X5778',
               'X5628NOHC', 'X5732ADHC']

path_label = join('/home/gyutaek/ssd1/Data/scRNA_seq_process_subtype', data_name)
path_output = join('/home/gyutaek/hdd2/Results/scMamba/2024-06-02-14-08-35-164237(celltype_jung_3cls_5epoch_lr1e-4_256dim)', data_name)

path_fig = join(path_output, 'fig')
makedirs(path_fig, exist_ok=True)

celltype_list = ['Astrocyte', 'Microglia', 'Oligodendrocyte', 'OPC', 'Excitatory', 'Inhibitory', 'Endothelial', 'Pericyte']
celltype_list_abbr = ['AC', 'MG', 'OL', 'OPC', 'EXN', 'INN', 'EC', 'PC']

table = np.zeros((len(celltype_list), len(celltype_list)))
truth = []
pred = []

for i in tqdm(range(len(test_donors))):
    path_sub = join(path_output, test_donors[i])
    list_cell = sorted(listdir(path_sub))

    for j in range(len(list_cell)):
        file_label = torch.load(join(path_label, test_donors[i], list_cell[j]))
        label = celltype_list.index(file_label['celltype'])
        file_output = torch.load(join(path_sub, list_cell[j]))
        output = celltype_list.index(file_output['prediction'])
        truth.append(celltype_list_abbr[label])
        pred.append(celltype_list_abbr[output])
        table[label, output] += 1

sns.set(font_scale=2)

cm = confusion_matrix(truth, pred, labels=celltype_list_abbr)
cm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
sns.heatmap(cm, ax=ax, xticklabels=celltype_list_abbr, yticklabels=celltype_list_abbr, cmap='viridis', vmin=0, vmax=1)
fig.savefig(path_fig + '/confusion_matrix.png', bbox_inches='tight', dpi=300)
