# scMamba: A Pre-Trained Model for Single-Nucleus RNA Sequencing Analysis in Neurodegenerative Disorders

This repository is the official pytorch implementation of "scMamba: A Pre-Trained Model for Single-Nucleus RNA Sequencing Analysis in Neurodegenerative Disorders".  
The code was modified from [[this repo]](https://github.com/HazyResearch/hyena-dna), and some parts were modified from [[this repo]](https://github.com/state-spaces/mamba)

## Requirements

The code is implented in Python 3.8 with below packages.
```
torch               1.13.0
pytorch-lightning   1.8.6
numpy               1.24.4
scipy               1.10.1
pandas              2.0.3
scanpy              1.9.3
anndata             0.8.0
```

## Checkpoint
If you want to use pre-trained model checkpoint, please contact [[okt0711@gmail.com]](okt0711@gmail.com).

## Usage
You can pre-train your model from scratch by running with:
```
python -m train wandb=null experiment=pretrain
```
You can fine-tune pre-trained model for downstream tasks by running with:
```
python -m train wandb=null experiment=[downstream task]
```
```[downstream task]``` can be ```celltype```, ```subtype```, ```subcluster```, ```doublet```, and ```imputation```.
