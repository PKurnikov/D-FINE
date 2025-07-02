"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class SharedSolver(BaseSolver):
    def is_better(self, current, best, mode='max'):
        if mode == 'max':
            return current > best
        else:
            return current < best

    def get_metric_from_stats(self, stats, dataloader_name, evaluator_name, metric_index):
        """Извлечь значение метрики из результата валидации."""
        try:
            obj = stats[dataloader_name][metric_index]
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            else:
                return obj
            # множественные evaluator_name
            # return stats[dataloader_name][evaluator_name][metric_index]
        except KeyError as e:
            raise KeyError(f"Missing key in test_stats: {e}")
        except IndexError as e:
            raise IndexError(f"Metric index out of range: {e}")

    def save_checkpoint(self, state_dict, output_dir, filename):
        dist_utils.save_on_master(state_dict, output_dir / filename)

    def serialize_stats(self, stats):
        def serialize_value(v):
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    return v.item()
                return v.tolist()  # превращает в list[float/int/...]
            elif isinstance(v, dict):
                return {k: serialize_value(sub_v) for k, sub_v in v.items()}
            else:
                return v

        return {k: serialize_value(v) for k, v in stats.items()}


    def evaluate_all(self, model, epoch):
        all_test_stats = {}
        all_evaluators = {}
        for idx, dataloader in enumerate(self.val_dataloaders):
            meta = getattr(dataloader, '_meta', {})
            key = meta.get('name')
            evaluator = self.evaluators.get(key, None)
            postprocessor = self.postprocessors.get(key, None)

            if evaluator is None:
                continue

            evaluator = evaluate(
                model, self.criterion, postprocessor,
                dataloader, evaluator, self.device,
                epoch, self.use_wandb, output_dir=self.output_dir
            )

            all_evaluators[key] = evaluator
            
        for key, evaluator_ in all_evaluators.items():
            if evaluator_.metric._update_called:
                stats = evaluator_.metric.compute()
                all_test_stats[key] = stats
                print(f"Evaluator '{key}':")
                print(evaluator_.summary())
            else:
                print(f"Rank {dist_utils.get_rank()} skipped compute — no updates were made.")

        return all_test_stats, all_evaluators

    def fit(self):
        self.train()
        args = self.cfg

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(args)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        best_criterion = -float('inf') if args.best_criterion['mode'] == 'max' else float('inf')
        best_epoch = -1

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            for dataloader in self.train_dataloaders:
                dataloader.set_epoch(epoch)

            if dist_utils.is_dist_available_and_initialized():
                for dataloader in self.train_dataloaders:
                    dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloaders,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            module = self.ema.module if self.ema else self.model
            test_stats, evaluators = self.evaluate_all(module, epoch=self.last_epoch)

            # Extract monitored metric for best model saving
            bm_cfg = args.best_criterion
            current_metric = self.get_metric_from_stats(
                test_stats,
                bm_cfg['dataloader'],
                bm_cfg['evaluator'],
                bm_cfg['metric_index']
            )

            if self.is_better(current_metric, best_criterion, bm_cfg['mode']):
                best_criterion = current_metric
                best_epoch = epoch
                stage = 'stg2' if epoch >= self.train_dataloaders[-1].collate_fn.stop_epoch else 'stg1'
                self.save_checkpoint(self.state_dict(), self.output_dir, f"best_{stage}.pth")
                print(f"New best model at epoch {epoch}: {best_criterion:.4f}")

            # Regular checkpoint saving
            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for path in checkpoint_paths:
                    self.save_checkpoint(self.state_dict(), self.output_dir, path.name)

            log_stats = {
                **{f"train_{k}": v for k, v in self.serialize_stats(train_stats).items()},
                **{f"test_{k}": v for k, v in self.serialize_stats(test_stats).items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
                'best_criterion': best_criterion,
                'best_epoch': best_epoch,
            }

            if self.use_wandb:
                wandb.log({"epoch": epoch, bm_cfg['evaluator']: current_metric})

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model

        all_test_stats, evaluators = self.evaluate_all(module, epoch=-1)

        return