import torch
from torch.utils.data import DataLoader

import os
import yaml 
from tqdm import tqdm

from src.models.mamba_ssm.models.mixer_seq_simple import MambaCLSHeadModel
from src.dataloaders.datasets.celltype_dataset import CelltypeDataset

from src.dataloaders.datasets.tokenizer import RNATokenizer

class CelltypeInference:
    def __init__(self, cfg):
        self.backbone, self.tokenizer = self.load_model(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        cfg['dataset']['tokenizer'] = self.tokenizer
        dataset = CelltypeDataset(**cfg['dataset'])
        self.dataloader = DataLoader(dataset,
                                     batch_size=cfg['batch_size'],
                                     num_workers=cfg['num_workers'],
                                     pin_memory=cfg['pin_memory'],
                                     shuffle=False,
                                     drop_last=False
                                     )
        
        self.output_path = cfg['output_path']

        cls_dict = dataset.cls_dict
        self.cls_dict = {v: k for k, v in cls_dict.items()}
    
    def load_model(self, cfg):
        model_cfg = cfg['model']

        backbone = MambaCLSHeadModel(**model_cfg)

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
        acc_sum = 0
        with tqdm(self.dataloader, unit='batch') as tloader:
            for expression, celltype, gene_ids, file_name in tloader:
                expression = expression.to(self.device)
                celltype = celltype.to(self.device)
                gene_ids = gene_ids.to(self.device)
                
                prediction = self.backbone(expression, gene_ids)[0]

                _, pred_class = prediction.max(1)

                for i in range(pred_class.size(0)):
                    pred = int(pred_class[i].detach().cpu().numpy())
                    pred = self.cls_dict[pred]

                    folder_name = os.path.join(self.output_path, file_name[0][i], file_name[1][i])
                    os.makedirs(folder_name, exist_ok=True)
                    torch.save({'prediction': pred}, os.path.join(folder_name, file_name[2][i]))
                
                acc_sum += (celltype == pred_class).sum()
                cnt += celltype.size(0)
                accuracy = acc_sum / cnt
                tloader.set_postfix(accuracy=accuracy.detach().cpu().numpy())


if __name__ == "__main__":
    cfg_path = 'configs/experiment/celltype_inference.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

    task = CelltypeInference(cfg)

    task.inference()