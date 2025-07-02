"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from pathlib import Path
from typing import Callable, Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    "BaseConfig",
]

BestCriterionConfig = Dict[str, Any]

class BaseConfig(object):
    # TODO property

    def __init__(self) -> None:
        super().__init__()

        self.task: str = None

        # instance / function
        self._model :nn.Module = None
        self._postprocessor :nn.Module = None
        self._criterion :nn.Module = None
        self._optimizer :Optimizer = None
        self._lr_scheduler :LRScheduler = None
        self._lr_warmup_scheduler: LRScheduler = None
        self._train_dataloaders :List[DataLoader] = None
        self._val_dataloaders :List[DataLoader] = None
        self._ema :nn.Module = None
        self._scaler :GradScaler = None
        self._train_datasets :List[Dataset] = None
        self._val_datasets :List[Dataset] = None
        self._collate_fn :Callable = None
        self._best_criterion: Optional[BestCriterionConfig] = None
        self._evaluators: Optional[Dict[str, Any]] = None
        self._postprocessors: Optional[Dict[str, Any]] = None

        self._writer: SummaryWriter = None

        # dataset
        self.num_workers: int = 0
        self.batch_size: int = None
        self._train_batch_size: int = None
        self._val_batch_size: int = None
        self._train_shuffle: bool = None
        self._val_shuffle: bool = None

        # runtime
        self.resume: str = None
        self.tuning: str = None

        self.epochs: int = None
        self.last_epoch: int = -1

        self.use_amp: bool = False
        self.use_ema: bool = False
        self.ema_decay: float = 0.9999
        self.ema_warmups: int = 2000
        self.sync_bn: bool = False
        self.clip_max_norm: float = 0.0
        self.find_unused_parameters: bool = None

        self.seed: int = None
        self.print_freq: int = None
        self.checkpoint_freq: int = 1
        self.output_dir: str = None
        self.summary_dir: str = None
        self.device: str = ""

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, m):
        assert isinstance(m, nn.Module), f"{type(m)} != nn.Module, please check your model class"
        self._model = m

    @property
    def postprocessor(self) -> nn.Module:
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, m):
        assert isinstance(m, nn.Module), f"{type(m)} != nn.Module, please check your model class"
        self._postprocessor = m

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @criterion.setter
    def criterion(self, m):
        assert isinstance(m, nn.Module), f"{type(m)} != nn.Module, please check your model class"
        self._criterion = m

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, m):
        assert isinstance(
            m, Optimizer
        ), f"{type(m)} != optim.Optimizer, please check your model class"
        self._optimizer = m

    @property
    def lr_scheduler(self) -> LRScheduler:
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, m):
        assert isinstance(
            m, LRScheduler
        ), f"{type(m)} != LRScheduler, please check your model class"
        self._lr_scheduler = m

    @property
    def lr_warmup_scheduler(self) -> LRScheduler:
        return self._lr_warmup_scheduler

    @lr_warmup_scheduler.setter
    def lr_warmup_scheduler(self, m):
        self._lr_warmup_scheduler = m

    @property
    def train_dataloaders(self) -> List[DataLoader]:
        if not self._train_dataloaders and self.train_datasets:
            self._train_dataloaders = [
                DataLoader(
                    ds,
                    batch_size=self.train_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                    shuffle=self.train_shuffle,
                )
                for ds in self.train_datasets
            ]
        return self._train_dataloaders

    @train_dataloaders.setter
    def train_dataloaders(self, loaders: List[DataLoader]):
        self._train_dataloaders = loaders

    @property
    def val_dataloaders(self) -> List[DataLoader]:
        if not self._val_dataloaders and self.val_datasets:
            self._val_dataloaders = [
                DataLoader(
                    ds,
                    batch_size=self.val_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                    shuffle=self.val_shuffle,
                )
                for ds in self.val_datasets
            ]
        return self._val_dataloaders

    @val_dataloaders.setter
    def val_dataloaders(self, loaders: List[DataLoader]):
        self._val_dataloaders = loaders

    @property
    def ema(self, ) -> nn.Module:
        if self._ema is None and self.use_ema and self.model is not None:
            from ..optim import ModelEMA

            self._ema = ModelEMA(self.model, self.ema_decay, self.ema_warmups)
        return self._ema

    @ema.setter
    def ema(self, obj):
        self._ema = obj

    @property
    def scaler(self) -> GradScaler:
        if self._scaler is None and self.use_amp and torch.cuda.is_available():
            self._scaler = GradScaler()
        return self._scaler

    @scaler.setter
    def scaler(self, obj: GradScaler):
        self._scaler = obj

    @property
    def val_shuffle(self) -> bool:
        if self._val_shuffle is None:
            print("warning: set default val_shuffle=False")
            return False
        return self._val_shuffle

    @val_shuffle.setter
    def val_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be bool"
        self._val_shuffle = shuffle

    @property
    def train_shuffle(self) -> bool:
        if self._train_shuffle is None:
            print("warning: set default train_shuffle=True")
            return True
        return self._train_shuffle

    @train_shuffle.setter
    def train_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be bool"
        self._train_shuffle = shuffle

    @property
    def train_batch_size(self) -> int:
        if self._train_batch_size is None and isinstance(self.batch_size, int):
            print(f"warning: set train_batch_size=batch_size={self.batch_size}")
            return self.batch_size
        return self._train_batch_size

    @train_batch_size.setter
    def train_batch_size(self, batch_size):
        assert isinstance(batch_size, int), "batch_size must be int"
        self._train_batch_size = batch_size

    @property
    def val_batch_size(self) -> int:
        if self._val_batch_size is None:
            print(f"warning: set val_batch_size=batch_size={self.batch_size}")
            return self.batch_size
        return self._val_batch_size

    @val_batch_size.setter
    def val_batch_size(self, batch_size):
        assert isinstance(batch_size, int), "batch_size must be int"
        self._val_batch_size = batch_size

    # @property
    # def train_dataset(self) -> Dataset:
    #     return self._train_dataset

    # @train_dataset.setter
    # def train_dataset(self, dataset):
    #     assert isinstance(dataset, Dataset), f"{type(dataset)} must be Dataset"
    #     self._train_dataset = dataset

    @property
    def train_datasets(self) -> List[Dataset]:
        return self._train_datasets

    @train_datasets.setter
    def train_datasets(self, datasets: List[Dataset]):
        assert all(isinstance(ds, Dataset) for ds in datasets), "All items must be torch Datasets"
        self._train_datasets = datasets

    @property
    def val_datasets(self) -> List[Dataset]:
        return self._val_datasets

    @val_datasets.setter
    def val_datasets(self, datasets: List[Dataset]):
        assert all(isinstance(ds, Dataset) for ds in datasets), "All items must be torch Datasets"
        self._val_datasets = datasets

    @property
    def collate_fn(self) -> Callable:
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, fn):
        assert isinstance(fn, Callable), f"{type(fn)} must be Callable"
        self._collate_fn = fn

    @property
    def evaluators(self) -> dict:
        return self._evaluators  # это словарь evaluators

    @evaluators.setter
    def evaluators(self, evaluators_dict):
        assert isinstance(evaluators_dict, dict), f"{type(evaluators_dict)} must be dict"
        self._evaluator = evaluators_dict

    # @property
    # def best_criterion(self) -> BestCriterionConfig:
    @property
    def best_criterion(self) -> BestCriterionConfig:
        if self._best_criterion is None:
            if "best_criterion" not in self.yaml_cfg:
                raise ValueError("Missing 'best_criterion' config in YAML.")
            cfg = self.yaml_cfg["best_criterion"]

            required_keys = ["dataloader", "evaluator", "metric_index", "mode"]
            for key in required_keys:
                if key not in cfg:
                    raise ValueError(f"Missing key '{key}' in best_criterion config.")

            if cfg["mode"] not in ["min", "max"]:
                raise ValueError("best_criterion 'mode' must be 'min' or 'max'.")

            self._best_criterion = cfg  # кэшируем результат
        return self._best_criterion

    @best_criterion.setter
    def best_criterion(self, value):
        # Валидация при ручной установке
        required_keys = ["dataloader", "evaluator", "metric_index", "mode"]
        for key in required_keys:
            if key not in value:
                raise ValueError(f"Missing key '{key}' in best_criterion config.")
        self._best_criterion = value

    @property
    def writer(self) -> SummaryWriter:
        if self._writer is None:
            if self.summary_dir:
                self._writer = SummaryWriter(self.summary_dir)
            elif self.output_dir:
                self._writer = SummaryWriter(Path(self.output_dir) / "summary")
        return self._writer

    @writer.setter
    def writer(self, m):
        assert isinstance(m, SummaryWriter), f"{type(m)} must be SummaryWriter"
        self._writer = m

    def __repr__(self):
        s = ""
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                s += f"{k}: {v}\n"
        return s
