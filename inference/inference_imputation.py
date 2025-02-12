import torch
from torch.utils.data import DataLoader

import os
import yaml
import scanpy as sc
from tqdm import tqdm

from src.models.mamba_ssm.models.mixer_seq_simple import MambaModel
from src.dataloaders.datasets.imputation_dataset import ImputationDataset
from src.tasks.metrics import mse

from src.dataloaders.datasets.tokenizer import RNATokenizer


def generate_zero_mask(expression):
    nB, seq_len = expression.size()
    mask = torch.zeros((10, nB, seq_len)).bool()
    for b in range(nB):
        zero_idx = torch.nonzero(expression[b] == 0)
        for i in range(10):
            sub_idx = zero_idx[i::10]
            mask[i, b, sub_idx] = True
    return mask


def generate_nonzero_mask(expression):
    nB, seq_len = expression.size()
    mask = torch.zeros((5, nB, seq_len)).bool()
    for b in range(nB):
        zero_idx = torch.nonzero(expression[b])
        for i in range(5):
            sub_idx = zero_idx[i::5]
            mask[i, b, sub_idx] = True
    return mask


class ImputationInference:
    def __init__(self, cfg):
        self.backbone, self.tokenizer = self.load_model(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        cfg['dataset']['tokenizer'] = self.tokenizer
        dataset = ImputationDataset(**cfg['dataset'], umi_filter=False)
        dataset_simul = ImputationDataset(**cfg['dataset'], umi_filter=True)
        self.dataloader = DataLoader(dataset,
                                     batch_size=cfg['batch_size'],
                                     num_workers=cfg['num_workers'],
                                     pin_memory=cfg['pin_memory'],
                                     shuffle=False,
                                     drop_last=False
                                     )
        self.dataloader_simul = DataLoader(dataset_simul,
                                           batch_size=cfg['batch_size'],
                                           num_workers=cfg['num_workers'],
                                           pin_memory=cfg['pin_memory'],
                                           shuffle=False,
                                           drop_last=False
                                           )
        
        self.output_path = cfg['output_path']
        self.h5ad_path = cfg['h5ad_path']
        self.studies = cfg['dataset']['studies']
        self.h5ad = {}
        self.h5ad_simul = {}
        for study in self.studies:
            data = sc.read_h5ad(os.path.join(self.h5ad_path, study, 'data.h5ad'))
            self.h5ad[study] = data.copy()
            data_simul = sc.read_h5ad(os.path.join((self.h5ad_path + '_UMI4000'), study, 'data.h5ad'))
            self.h5ad_simul[study] = data_simul.copy()
    
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
        cnt = 0
        loss_sum = 0
        with tqdm(self.dataloader, unit='batch') as tloader:
            for input_exp, target_exp, gene_ids, file_name in tloader:
                input_exp = input_exp.to(self.device)
                target_exp = target_exp.to(self.device)
                gene_ids = gene_ids.to(self.device)

                masked_indices_zero = generate_zero_mask(input_exp)
                masked_indices_nonzero = generate_nonzero_mask(input_exp)

                imputed_exp = target_exp.clone()
                for i in range(10):
                    masked_exp = input_exp.clone()
                    masked_exp[masked_indices_zero[i]] = 0.0
                    prediction = self.backbone(masked_exp, gene_ids, masked_indices_zero[i].to(self.device))[0]
                    imputed_exp[masked_indices_zero[i]] = prediction[masked_indices_zero[i]]
                
                imputed_exp_simul = torch.zeros_like(target_exp)
                for i in range(5):
                    masked_exp = input_exp.clone()
                    masked_exp[masked_indices_nonzero[i]] = 0.0
                    prediction = self.backbone(input_exp, gene_ids, masked_indices_nonzero[i].to(self.device))[0]
                    imputed_exp_simul[masked_indices_nonzero[i]] = prediction[masked_indices_nonzero[i]]

                loss_sum += mse(imputed_exp_simul, target_exp, ignore_index=0)
                cnt += 1
                loss = loss_sum / cnt
                tloader.set_postfix(loss=loss.detach().cpu().numpy())
                
                imputed_exp = imputed_exp.detach().cpu().numpy()
                imputed_exp_simul = imputed_exp_simul.detach().cpu().numpy()
                
                for i in range(input_exp.size(0)):
                    self.h5ad[file_name[0][i]].X[self.h5ad[file_name[0][i]].obs_names.tolist().index(file_name[2][i])] = imputed_exp[i]
                    self.h5ad_simul[file_name[0][i]].X[self.h5ad_simul[file_name[0][i]].obs_names.tolist().index(file_name[2][i])] = imputed_exp_simul[i]
                
        for study in self.studies:
            os.makedirs(os.path.join(self.output_path, study), exist_ok=True)
            self.h5ad[study].write(os.path.join(self.output_path, study, 'imputed.h5ad'))
            self.h5ad_simul[study].write(os.path.join(self.output_path, study, 'imputed_simul.h5ad'))


if __name__ == "__main__":
    cfg_path = 'configs/experiment/imputation_inference.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

    task = ImputationInference(cfg)

    task.inference()