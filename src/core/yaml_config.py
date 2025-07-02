"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
import re
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mmengine.config import Config

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict, flatten_global_cfg

class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        # cfg = load_config(cfg_path)
        cfg = Config.fromfile(cfg_path)
        cfg = merge_dict(cfg.to_dict(), kwargs)
        # List[Dict] регистрирует как ключ name - значение dict()
        cfg = flatten_global_cfg(cfg)
        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith("_") and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None and "model" in self.yaml_cfg:
            self._model = create(self.yaml_cfg["model"], self.global_cfg)
        return super().model

    # @property
    # def postprocessor(self) -> torch.nn.Module:
    #     if self._postprocessor is None and "postprocessor" in self.yaml_cfg:
    #         self._postprocessor = create(self.yaml_cfg["postprocessor"], self.global_cfg)
    #     return super().postprocessor

    @property
    def postprocessor(self) -> torch.nn.Module:
        if self._postprocessor is None:
            cfg = self.yaml_cfg.get("postprocessor", None)
            assert cfg is not None, "Missing 'postprocessor' config"
            assert isinstance(cfg, dict) and "type" in cfg, \
                "Expected 'postprocessor' to be a dict with 'type' key"
            self._postprocessor = create('postprocessor', self.global_cfg)
        return self._postprocessor

    @property
    def criterion(self) -> torch.nn.Module:
        if self._criterion is None and "criterion" in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg["criterion"], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None and "optimizer" in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg["optimizer"], self.model)
            self._optimizer = create("optimizer", self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and "lr_scheduler" in self.yaml_cfg:
            self._lr_scheduler = create("lr_scheduler", self.global_cfg, optimizer=self.optimizer)
            print(f"Initial lr: {self._lr_scheduler.get_last_lr()}")
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and "lr_warmup_scheduler" in self.yaml_cfg:
            self._lr_warmup_scheduler = create(
                "lr_warmup_scheduler", self.global_cfg, lr_scheduler=self.lr_scheduler
            )
        return super().lr_warmup_scheduler

    @property
    def train_dataloaders(self) -> List[DataLoader]:
        if self._train_dataloaders is None:
            self._train_dataloaders = []

            dataloader_cfgs = self.yaml_cfg.get('dataloaders', [])
            for dl_cfg in dataloader_cfgs:
                if dl_cfg.get('role') == 'train':
                    name = dl_cfg.get('name')
                    dataloader = self.build_dataloader(name)
                    self._train_dataloaders.append(dataloader)

        return super().train_dataloaders

    @property
    def val_dataloaders(self) -> List[DataLoader]:
        if self._val_dataloaders is None:
            self._val_dataloaders = []

            dataloader_cfgs = self.yaml_cfg.get('dataloaders', [])
            for dl_cfg in dataloader_cfgs:
                if dl_cfg.get('role') == 'val':
                    name = dl_cfg.get('name')
                    dataloader = self.build_dataloader(name)
                    self._val_dataloaders.append(dataloader)

        return super().val_dataloaders

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self):
        if self._scaler is None and self.yaml_cfg.get("use_amp", False):
            self._scaler = create("scaler", self.global_cfg)
        return super().scaler

    @property
    def evaluators(self):
        if self._evaluators is None:
            self._evaluators = {}
            self._postprocessors = {}

            for loader in self.val_dataloaders:
                meta = getattr(loader, '_meta', {})
                name = meta.get('name')

                # Найти конкретный dataloader config по имени
                loader_cfg = next(
                    (cfg for cfg in self.global_cfg['dataloaders'] if cfg['name'] == name),
                    None
                )
                if loader_cfg is None:
                    continue

                # binding evaluator и postprocessor
                evaluator, postprocessor = self.build_evaluator_and_postprocessor(loader_cfg)
                self._evaluators[name] = evaluator
                self._postprocessors[name] = postprocessor

        return self._evaluators

    @property
    def postprocessors(self):
        if self._postprocessors is None:
            _ = self.evaluators  # вызываем evaluators чтобы заполнить постпроцессоры тоже
        return self._postprocessors

    @property
    def use_wandb(self) -> bool:
        return self.yaml_cfg.get("use_wandb", False)

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert "type" in cfg, ""
        cfg = copy.deepcopy(cfg)

        if "params" not in cfg:
            return model.parameters()

        assert isinstance(cfg["params"], list), ""

        param_groups = []
        visited = []
        for pg in cfg["params"]:
            pattern = pg["params"]
            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad and len(re.findall(pattern, k)) > 0
            }
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ""

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided."""
        assert ("total_batch_size" in cfg or "batch_size" in cfg) and not (
            "total_batch_size" in cfg and "batch_size" in cfg
        ), "`batch_size` or `total_batch_size` should be choosed one"

        total_batch_size = cfg.get("total_batch_size", None)
        if total_batch_size is None:
            bs = cfg.get("batch_size")
        else:
            from ..misc import dist_utils

            assert (
                total_batch_size % dist_utils.get_world_size() == 0
            ), "total_batch_size should be divisible by world size"
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if "total_batch_size" in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop("total_batch_size")
        print(f"building {name} with batch_size={bs}...")
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get("shuffle", False)


        # Найти конкретный dataloader config по имени
        dataloader_cfg = next(
            (cfg for cfg in global_cfg['dataloaders'] if cfg['name'] == name),
            None
        )

        if dataloader_cfg is not None:
            name = dataloader_cfg['name']
            role = dataloader_cfg['role']
            evaluator = None
            postprocessor = None
            if 'evaluator' in dataloader_cfg:
                evaluator = dataloader_cfg['evaluator']
            if 'postprocessor' in dataloader_cfg:
                postprocessor = dataloader_cfg['postprocessor']
            # Метаинформация для инициализации evaluators и postprocessors
            loader._meta = {
                'name': name,
                'role': role,
                'evaluator': evaluator,
                'postprocessor': postprocessor
            }

        return loader
    
    def build_evaluator_and_postprocessor(self, cfg):
        evaluator_cfg = cfg.get('evaluator')
        evaluator = None
        if evaluator_cfg:
            evaluator_cfg_ = {'evaluator': evaluator_cfg}
            evaluator_cfg_.update({evaluator_cfg['type']: self.global_cfg[evaluator_cfg['type']]})
            evaluator = create('evaluator', evaluator_cfg_) if evaluator_cfg_ else None
        postprocessor_cfg = cfg.get('postprocessor')
        
        postprocessor = None
        if postprocessor_cfg:
            postprocessor_cfg_ = {'postprocessor': postprocessor_cfg}
            postprocessor_cfg_.update({postprocessor_cfg['type']: self.global_cfg[postprocessor_cfg['type']]})
            postprocessor = create('postprocessor', postprocessor_cfg_) if postprocessor_cfg_ else None

        return evaluator, postprocessor