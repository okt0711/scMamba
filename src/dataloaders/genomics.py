import os
import json
from pathlib import Path
from typing import Any, List, Union
from torch.utils.data.dataloader import DataLoader, Dataset
from datasets import Dataset

from src.dataloaders.base import SequenceDataset
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.dataloaders.datasets.tokenizer import RNATokenizer
from src.dataloaders.datasets.pretrain_dataset import PretrainDataset
from src.dataloaders.datasets.doublet_dataset import DoubletDataset
from src.dataloaders.datasets.celltype_dataset import CelltypeDataset
from src.dataloaders.datasets.subtype_dataset import SubtypeDataset
from src.dataloaders.datasets.subcluster_dataset import SubclusterDataset
from src.dataloaders.datasets.imputation_dataset import ImputationDataset


class Pretrain(SequenceDataset):
    _name_ = "pretrain"

    def __init__(self, studies, study_dir, mem_probability, dataset_config_name=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, return_embeds=False, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.studies = studies
        self.study_dir = study_dir
        self.vocab = Path(study_dir) / 'vocab.txt'
        self.mem_probability = mem_probability
        self.return_embeds = return_embeds

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant
    
    def setup(self, stage=None):
        self.tokenizer = RNATokenizer(vocab_file=self.vocab)

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()
    
    def init_datasets(self):
        self.dataset_train, self.dataset_val, self.dataset_test = [
            PretrainDataset(split=split,
                            studies=self.studies,
                            study_dir=self.study_dir,
                            mem_probability=self.mem_probability,
                            tokenizer=self.tokenizer,
                            add_eos=self.add_eos,
                            return_embeds=self.return_embeds,
                            )
            for split in ['train', 'valid', 'test']
        ]
        return
    
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)
        # return self._data_loader(self.dataset_train, batch_size=1,
        #                          shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
        # return self._data_loader(self.dataset_val, batch_size=1)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet


class Doublet(SequenceDataset):
    _name_ = "doublet"

    def __init__(self, studies, study_dir, cls_key, dataset_config_name=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, add_cls=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.add_cls = add_cls
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.studies = studies
        self.study_dir = study_dir
        self.vocab = Path(study_dir) / 'vocab.txt'
        self.cls_key = cls_key
        self.cls_dict = json.load(open(os.path.join(self.study_dir, 'doublet.json')))

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant
    
    def setup(self, stage=None):
        self.tokenizer = RNATokenizer(vocab_file=self.vocab)

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()
    
    def init_datasets(self):
        self.dataset_train, self.dataset_val, self.dataset_test = [
            DoubletDataset(split=split,
                           studies=self.studies,
                           study_dir=self.study_dir,
                           cls_key=self.cls_key,
                           tokenizer=self.tokenizer,
                           add_eos=self.add_eos,
                           add_cls=self.add_cls,
                           )
            for split in ['train', 'valid', 'test']
        ]
        return
    
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)
        # return self._data_loader(self.dataset_train, batch_size=1,
        #                          shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
        # return self._data_loader(self.dataset_val, batch_size=1)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet


class Celltype(SequenceDataset):
    _name_ = "celltype"

    def __init__(self, studies, study_dir, cls_key, dataset_config_name=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, add_cls=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.add_cls = add_cls
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.studies = studies
        self.study_dir = study_dir
        self.vocab = Path(study_dir) / 'vocab.txt'
        self.cls_key = cls_key
        self.cls_dict = json.load(open(os.path.join(self.study_dir, self.cls_key + '.json')))

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant
    
    def setup(self, stage=None):
        self.tokenizer = RNATokenizer(vocab_file=self.vocab)

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()
    
    def init_datasets(self):
        self.dataset_train, self.dataset_val, self.dataset_test = [
            CelltypeDataset(split=split,
                            studies=self.studies,
                            study_dir=self.study_dir,
                            cls_key=self.cls_key,
                            tokenizer=self.tokenizer,
                            add_eos=self.add_eos,
                            add_cls=self.add_cls,
                            )
            for split in ['train', 'valid', 'test']
        ]
        return
    
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)
        # return self._data_loader(self.dataset_train, batch_size=1,
        #                          shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
        # return self._data_loader(self.dataset_val, batch_size=1)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet


class Subtype(SequenceDataset):
    _name_ = "subtype"

    def __init__(self, studies, study_dir, dataset_config_name=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, add_cls=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.add_cls = add_cls
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.studies = studies
        self.study_dir = study_dir
        self.vocab = Path(study_dir) / 'vocab.txt'
        if 'Leng_2021_Nat_Neurosci' in self.studies:
            self.cls_dict = json.load(open(os.path.join(self.study_dir, 'subtype_ec.json')))
        else:
            self.cls_dict = json.load(open(os.path.join(self.study_dir, 'subtype.json')))

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant
    
    def setup(self, stage=None):
        self.tokenizer = RNATokenizer(vocab_file=self.vocab)

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()
    
    def init_datasets(self):
        self.dataset_train, self.dataset_val, self.dataset_test = [
            SubtypeDataset(split=split,
                           studies=self.studies,
                           study_dir=self.study_dir,
                           tokenizer=self.tokenizer,
                           add_eos=self.add_eos,
                           add_cls=self.add_cls,
                          )
            for split in ['train', 'valid', 'test']
        ]
        return
    
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)
        # return self._data_loader(self.dataset_train, batch_size=1,
        #                          shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
        # return self._data_loader(self.dataset_val, batch_size=1)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet


class Subcluster(SequenceDataset):
    _name_ = "subcluster"

    def __init__(self, studies, study_dir, dataset_config_name=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, add_cls=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.add_cls = add_cls
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.studies = studies
        self.study_dir = study_dir
        self.vocab = Path(study_dir) / 'vocab.txt'
        if 'Leng_2021_Nat_Neurosci' in self.studies:
            self.cls_dict = json.load(open(os.path.join(self.study_dir, 'subcluster_ec.json')))
        else:
            self.cls_dict = json.load(open(os.path.join(self.study_dir, 'subcluster.json')))

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant
    
    def setup(self, stage=None):
        self.tokenizer = RNATokenizer(vocab_file=self.vocab)

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()
    
    def init_datasets(self):
        self.dataset_train, self.dataset_val, self.dataset_test = [
            SubclusterDataset(split=split,
                              studies=self.studies,
                              study_dir=self.study_dir,
                              tokenizer=self.tokenizer,
                              add_eos=self.add_eos,
                              add_cls=self.add_cls,
                              )
            for split in ['train', 'valid', 'test']
        ]
        return
    
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)
        # return self._data_loader(self.dataset_train, batch_size=1,
        #                          shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
        # return self._data_loader(self.dataset_val, batch_size=1)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet


class Imputation(SequenceDataset):
    _name_ = "imputation"

    def __init__(self, studies, study_dir, mem_probability, umi_filter, dataset_config_name=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=True, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.studies = studies
        self.study_dir = study_dir
        self.vocab = Path(study_dir) / 'vocab.txt'
        self.mem_probability = mem_probability
        self.umi_filter = umi_filter

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant
    
    def setup(self, stage=None):
        self.tokenizer = RNATokenizer(vocab_file=self.vocab)

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()
    
    def init_datasets(self):
        self.dataset_train, self.dataset_val, self.dataset_test = [
            ImputationDataset(split=split,
                              studies=self.studies,
                              study_dir=self.study_dir,
                              mem_probability=self.mem_probability,
                              umi_filter=self.umi_filter,
                              tokenizer=self.tokenizer,
                              add_eos=self.add_eos,
                              )
            for split in ['train', 'valid', 'test']
        ]
        return
    
    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (FaultTolerantDistributedSampler(self.dataset_train,
                                                       **self.trainer.distributed_sampler_kwargs)
                       if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)
        # return self._data_loader(self.dataset_train, batch_size=1,
        #                          shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)
        # return self._data_loader(self.dataset_val, batch_size=1)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet
