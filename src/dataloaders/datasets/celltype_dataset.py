import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import torch


class CelltypeDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            studies: list,
            study_dir: str,
            split: str,
            cls_key: str,
            tokenizer=None,
            add_eos=False,
            add_cls=True,
    ):
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.add_cls = add_cls
        self.studies = studies
        self.study_dir = study_dir
        self.split = split
        self.cls_key = cls_key

        self.cells = []
        self.cls_dict = json.load(open(os.path.join(self.study_dir, self.cls_key + '.json')))
        
        # Load cell list and metadata
        for study in self.studies:
            study_path = Path(self.study_dir) / study
            assert study_path.exists(), f'The path `{study_path}` does not exist for study `{study}`. Please point to a valid directory containing your studies.'

            # Filter out doublets, unidentified, and minor type cells
            metadata = pd.read_csv(os.path.join(study_path, 'metadata.csv'), index_col=0)
            exclude_celltypes = ['Double_negative_Neuron',
                                 'Double_positive_Neuron',
                                 'Doublet',
                                 'GNLY_CD44_myeloid_sub1',
                                 'Unidentified']
            metadata = metadata[~metadata['celltype'].isin(exclude_celltypes)]
            
            batches = json.load(open(os.path.join(study_path, 'splits.json')))
            batches = batches[split]

            for batch in batches:
                metadata_batch = metadata[metadata['donor'] == batch]

                self.cells.extend([{'cell': cell + '.pt', 'batch': batch, 'path': study_path} for cell in metadata_batch.index])
        
    def __len__(self):
        return len(self.cells)
    
    def __getitem__(self, idx):
        # Load cell data
        cell = self.cells[idx]
        data = torch.load(os.path.join(cell['path'], cell['batch'], cell['cell']))

        # scRNA-seq expression data
        expression = torch.FloatTensor(data['RNA'])

        # Load gene list
        list_gene = open(os.path.join(cell['path'], 'vocab.txt'))
        gene_ids = list_gene.readlines()
        for i in range(len(gene_ids)):
            gene_ids[i] = gene_ids[i].strip()
        list_gene.close()
        gene_ids = np.array(gene_ids)
        
        gene_ids = ' '.join(gene_ids)

        # Tokenize gene list
        gene_ids = self.tokenizer(gene_ids,
                                  padding='longest',
                                  add_special_tokens=False)
        
        gene_ids = gene_ids["input_ids"]

        if self.add_eos:
            gene_ids.append(self.tokenizer.sep_token_id)
        # Add CLS token (head, middle, tail)
        if self.add_cls:
            len_gene = len(gene_ids)
            gene_ids.insert(len_gene // 2, self.tokenizer.cls_token_id)
            gene_ids.insert(0, self.tokenizer.cls_token_id)
            gene_ids.append(self.tokenizer.cls_token_id)
        
        gene_ids = torch.LongTensor(gene_ids)

        celltype = self.cls_dict[data[self.cls_key]]

        # If test split, return expression data and metadata
        if self.split == 'test':
            study = str(cell['path']).split('/')[-1]
            return expression, celltype, gene_ids, [study, cell['batch'], cell['cell']]
        
        return expression, celltype, {'gene_ids': gene_ids}
