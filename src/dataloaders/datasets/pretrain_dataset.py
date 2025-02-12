import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import torch


class PretrainDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            studies: list,
            study_dir: str,
            split: str,
            mem_probability: float,
            tokenizer=None,
            add_eos=False,
            return_embeds=False
    ):
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.studies = studies
        self.study_dir = study_dir
        self.split = split
        self.mem_probability = mem_probability
        self.return_embeds = return_embeds

        self.cells = []
        
        # Load cell list and metadata
        for study in self.studies:
            study_path = Path(self.study_dir) / study
            assert study_path.exists(), f'The path `{study_path}` does not exist for study `{study}`. Please point to a valid directory containing your studies.'

            # Filter out doublets and unidentified cells
            metadata = pd.read_csv(os.path.join(study_path, 'metadata.csv'), index_col=0)
            exclude_celltypes = ['Doublet',
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
        
        gene_ids = torch.LongTensor(gene_ids)

        # If test split or return_embeds is True, return the non-masked expression data and metadata
        if self.split == 'test' or self.return_embeds:
            study = str(cell['path']).split('/')[-1]
            return expression, gene_ids, data['celltype'], data['disease'], [study, cell['batch'], cell['cell'].split('.')[0]]

        # Mask expression data
        input_exp = expression.clone()
        target_exp = expression.clone()

        probability_matrix = torch.full(target_exp.shape, self.mem_probability)
        input_exp, masked_indices, target_exp = self.mask(input_exp, gene_ids, self.tokenizer, targets=target_exp,
                                                          probability_matrix=probability_matrix)

        return input_exp, target_exp, {'gene_ids': gene_ids, 'masked_indices': masked_indices}
    
    def mask(self, input_exp, input_ids, tokenizer, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Masked indices should not include padding tokens and zero values
        masked_indices[input_ids == tokenizer.pad_token_id] = False
        masked_indices[input_exp == 0.0] = False

        if targets is not None:
            targets[~masked_indices] = -100
        
        # Mask input expression data by setting masked indices to zero
        input_exp[masked_indices] = 0.0

        if targets is not None:
            return input_exp, masked_indices, targets
        else:
            return input_exp, masked_indices
