import torch
from torch.utils.data import DataLoader

import os
import yaml
import scanpy as sc
from anndata import AnnData
from tqdm import tqdm

from src.models.mamba_ssm.models.mixer_seq_simple import MambaModel
from src.dataloaders.datasets.pretrain_dataset import PretrainDataset
import numpy as np

from src.dataloaders.datasets.tokenizer import RNATokenizer


class PretrainEmbedding:
    def __init__(self, cfg):
        self.backbone, self.tokenizer = self.load_model(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        cfg['dataset']['tokenizer'] = self.tokenizer
        dataset = PretrainDataset(**cfg['dataset'])
        self.dataloader = DataLoader(dataset,
                                     batch_size=cfg['batch_size'],
                                     num_workers=cfg['num_workers'],
                                     pin_memory=cfg['pin_memory'],
                                     shuffle=False,
                                     drop_last=False
                                     )
        
        self.output_path = cfg['output_path']
        self.h5ad_path = cfg['h5ad_path']
        self.studies = cfg['dataset']['studies']
        if self.h5ad_path is not None:
            self.h5ad = {}
            for study in self.studies:
                data = sc.read_h5ad(os.path.join(self.h5ad_path, study, 'data.h5ad'))
                embedding = AnnData(X=np.zeros((data.shape[0], cfg['model']['d_model'])))
                embedding.obs = data.obs
                self.h5ad[study] = embedding
    
    def load_model(self, cfg):
        model_cfg = cfg['model']

        backbone = MambaModel(**model_cfg)

        state_dict = torch.load(cfg['ckpt_path'], map_location='cpu')

        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )

        model_state_dict = state_dict["state_dict"]

        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)
        
        backbone.load_state_dict(model_state_dict, strict=True)

        tokenizer = RNATokenizer(vocab_file=cfg['vocab_path'])

        return backbone, tokenizer
    
    @torch.no_grad()
    def inference(self):
        with tqdm(self.dataloader, unit='batch') as tloader:
            for input_exp, gene_ids, celltype, disease, file_name in tloader:
                input_exp = input_exp.to(self.device)
                gene_ids = gene_ids.to(self.device)

                embedding = self.backbone(input_exp, gene_ids, return_embeds=True)[0]
                embedding = torch.mean(embedding, dim=1)
                embedding = embedding.detach().cpu().numpy()

                for i in range(input_exp.size(0)):
                    if self.h5ad_path is not None:
                        self.h5ad[file_name[0][i]].X[self.h5ad[file_name[0][i]].obs_names.tolist().index(file_name[2][i])] = embedding[i]

                    folder_name = os.path.join(self.output_path, file_name[0][i], file_name[1][i])
                    os.makedirs(folder_name, exist_ok=True)
                    torch.save({'embedding': embedding[i], 'celltype': celltype[i], 'disease': disease[i]}, os.path.join(folder_name, file_name[2][i]+'.pt'))

        if self.h5ad_path is not None:
            for study in self.studies:
                os.makedirs(os.path.join(self.output_path, study), exist_ok=True)
                self.h5ad[study].write(os.path.join(self.output_path, study, 'embedding.h5ad'))


if __name__ == "__main__":
    cfg_path = 'configs/experiment/pretrain_embedding.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

    task = PretrainEmbedding(cfg)

    task.inference()