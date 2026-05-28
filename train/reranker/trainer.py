from __future__ import annotations


import os
import re
import resource
import shutil
import subprocess
from numbers import Number
from typing import Any, Sized

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        accelerator: Accelerator,
        validation_dataloader: DataLoader | None = None,
        epochs: int = 3,
        lr_scheduler: LRScheduler,
        log_interval: int = 10,
        save_on_epoch_end: bool = True,
        tokenizer,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end
        self.tokenizer = tokenizer
        self.min_val_loss = 100000
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.best_val_loss: float | None = None
        self.epochs_without_improvement = 0

        self.train_loss_tracker = LossTracker()
        self.validation_loss_tracker = LossTracker()
        if isinstance(self.train_dataloader.dataset, Sized):
            num_steps_per_epoch = len(self.train_dataloader)
        else:
            num_steps_per_epoch = None
        self.progress_bar = DistributedTqdmProgressBar(
            self.accelerator, self.epochs, num_steps_per_epoch=num_steps_per_epoch
        )
        self.current_step = 0

    def _run_model(self, batch):
        if isinstance(batch, dict):
            return self.model(batch)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return self.model(batch[0], batch[1])
        raise ValueError("Unsupported batch format")

    def train(self):
        for current_epoch in range(1, self.epochs + 1):
            self.model.train()
            self.progress_bar.on_epoch_start()

            for batch_index, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()

                    batch_output = self._run_model(batch)
                    loss = batch_output['loss']

                    self.accelerator.backward(loss)
                    # if batch_index % self.log_interval == 0:
                    #     # 仅适用于 zero 0/1，不适用于 zero 2/3、FSDP 等梯度分片存储的情况
                    #     total_norm = torch.nn.utils.clip_grad_norm_(
                    #         self.model.parameters(), 
                    #         max_norm=float('inf'), # 设为无穷大以仅计算范数，不实际裁剪
                    #         norm_type=2
                    #     )
                    
                    self.optimizer.step()

                    self.lr_scheduler.step()
                    self.train_loss_tracker.update(loss)

                if batch_index % self.log_interval == 0:
                    loss_breakdown = {
                        key: value
                        for key, value in batch_output.get("losses", {}).items()
                        if key != "loss"
                    }
                    log_dic = {
                        'step_train_loss': batch_output['loss'],
                        'lr': float(self.lr_scheduler.get_last_lr()[0]),
                        'train_loss': self.train_loss_tracker.loss,
                    }
                    log_dic.update(loss_breakdown)
                    self.log_metrics(
                        log_dic,
                        step=self.current_step,
                    )

                if (
                    self.validation_dataloader
                    and batch_index % (self.log_interval * 10) == 0
                ):
                    validation_loss = evaluate(
                        self.model,
                        self.validation_dataloader,
                        self.validation_loss_tracker,
                    )
                    if isinstance(validation_loss, dict):
                        self.accelerator.log(validation_loss, step=self.current_step)
                    else:
                        self.accelerator.log(
                            {"valid_loss": validation_loss}, step=self.current_step
                        )
                    # If you want to save the model with min validation loss, uncomment the following code.
                    # if validation_loss < self.min_val_loss:
                    #     if self.accelerator.is_local_main_process and self.current_step > 0:
                    #         save_dir = self.get_checkpoint_dir(current_epoch)
                    #         save_dir = os.path.join(
                    #             save_dir,
                    #             f"_min_val_loss",
                    #         )
                    #         self.min_val_loss = validation_loss
                    #         print(f"Saving model with min validation loss: {validation_loss}, step: {self.current_step}")
                    #         unwrapped_model = self.accelerator.unwrap_model(self.model)
                    #         unwrapped_model.save_pretrained(
                    #             save_dir, safe_serialization=True
                    #         )
                    #         self.tokenizer.save_pretrained(save_dir)
                    #     self.accelerator.wait_for_everyone()

                self.progress_bar.update()
                self.current_step += 1

            epoch_validation_metrics = None
            if self.validation_dataloader:
                epoch_validation_metrics = evaluate(
                    self.model,
                    self.validation_dataloader,
                    self.validation_loss_tracker,
                )
                if isinstance(epoch_validation_metrics, dict):
                    self.accelerator.log(epoch_validation_metrics, step=self.current_step)
                else:
                    self.accelerator.log(
                        {"valid_loss": epoch_validation_metrics}, step=self.current_step
                    )

            self.train_loss_tracker.on_epoch_end()
            self.progress_bar.on_epoch_end()
            self._log_epoch_resource_usage(current_epoch)

            if self.save_on_epoch_end:
                if self.accelerator.is_local_main_process:
                    save_dir = self.get_checkpoint_dir(current_epoch)
                    print(save_dir)
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    unwrapped_model.save_pretrained(save_dir, safe_serialization=False)
                    self.tokenizer.save_pretrained(save_dir)
                self.accelerator.wait_for_everyone()

            if self._should_stop_early(epoch_validation_metrics):
                if self.accelerator.is_main_process:
                    self.accelerator.print(
                        "Early stopping triggered: validation loss did not improve "
                        f"for {self.early_stopping_patience} epoch(s)."
                    )
                break

        self.accelerator.end_training()

    def _should_stop_early(self, validation_metrics) -> bool:
        if self.early_stopping_patience is None or validation_metrics is None:
            return False

        if isinstance(validation_metrics, dict):
            current_val_loss = validation_metrics.get("valid_loss")
        else:
            current_val_loss = validation_metrics

        if current_val_loss is None:
            return False

        current_val_loss = float(current_val_loss)
        if self.best_val_loss is None or (
            self.best_val_loss - current_val_loss
        ) > self.early_stopping_min_delta:
            self.best_val_loss = current_val_loss
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.early_stopping_patience

    def log_metrics(self, metrics: dict[str, float], step: int):
        metrics = {
            key: _to_loggable_scalar(value)
            for key, value in metrics.items()
        }
        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f"{prefix}/{k}": v for k, v in values.items()}

    def get_checkpoint_dir(self, current_epoch):

        self.accelerator.project_configuration.automatic_checkpoint_naming = False
        output_dir = os.path.join(self.accelerator.project_dir, "checkpoints")
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)
            folders = [
                os.path.join(output_dir, folder) for folder in os.listdir(output_dir)
            ]
            if self.accelerator.project_configuration.total_limit is not None and (
                len(folders) + 1 > self.accelerator.project_configuration.total_limit
            ):

                def _inner(folder):
                    return list(
                        map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder))
                    )[0]

                folders.sort(key=_inner)
                for folder in folders[
                    : len(folders)
                    + 1
                    - self.accelerator.project_configuration.total_limit
                ]:
                    shutil.rmtree(folder)

        output_dir = os.path.join(output_dir, f"checkpoint_{current_epoch-1}")
        if self.accelerator.is_local_main_process:
            os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _log_epoch_resource_usage(self, current_epoch: int) -> None:
        metrics = _collect_resource_metrics(self.accelerator)
        if not metrics:
            return

        self.accelerator.log(metrics, step=self.current_step)
        if self.accelerator.is_main_process:
            summary = ", ".join(f"{name}={value:.2f}" for name, value in metrics.items())
            self.accelerator.print(f"Epoch {current_epoch} resource usage: {summary}")


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_tracker: LossTracker | None = None,
):
    loss_tracker = loss_tracker or LossTracker()
    metric_trackers = {}
    for batch in dataloader:
        with torch.inference_mode():
            if isinstance(batch, dict):
                batch_output = model(batch)
            else:
                batch_output = model(batch[0], batch[1])
            loss_tracker.update(batch_output['loss'])
            for key, value in batch_output.get("losses", {}).items():
                tracker = metric_trackers.setdefault(key, LossTracker())
                tracker.update(value)
    loss = loss_tracker.loss
    loss_tracker.on_epoch_end()
    if metric_trackers:
        metrics = {
            "valid_loss": loss,
            **{
                f"valid_{key}": tracker.loss
                for key, tracker in metric_trackers.items()
                if key != "loss"
            },
        }
        for tracker in metric_trackers.values():
            tracker.on_epoch_end()
        return metrics
    return loss


def _to_loggable_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().cpu().mean().item())
    if isinstance(value, Number):
        return float(value)
    return value


def _collect_resource_metrics(accelerator: Accelerator) -> dict[str, float]:
    metrics = {
        "epoch/process_max_rss_mb": _get_process_rss_mb(),
    }
    metrics.update(_get_gpu_resource_metrics(accelerator))
    return metrics


def _get_process_rss_mb() -> float:
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if os.uname().sysname == "Darwin":
        return rss_kb / (1024 * 1024)
    return rss_kb / 1024


def _get_gpu_resource_metrics(accelerator: Accelerator) -> dict[str, float]:
    metrics: dict[str, float] = {}

    if torch.cuda.is_available():
        device = accelerator.device
        if device.type == "cuda":
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            metrics["epoch/gpu_mem_allocated_mb"] = (
                torch.cuda.memory_allocated(device_index) / (1024 ** 2)
            )
            metrics["epoch/gpu_mem_reserved_mb"] = (
                torch.cuda.memory_reserved(device_index) / (1024 ** 2)
            )
            metrics["epoch/gpu_mem_max_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
            )

            nvidia_smi_metrics = _query_nvidia_smi(device_index)
            metrics.update(nvidia_smi_metrics)

    return metrics


def _query_nvidia_smi(device_index: int) -> dict[str, float]:
    if shutil.which("nvidia-smi") is None:
        return {}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {}

    line = result.stdout.strip().splitlines()
    if not line:
        return {}

    try:
        gpu_util, mem_used_mb, mem_total_mb = [
            float(value.strip()) for value in line[0].split(",")
        ]
    except Exception:
        return {}

    metrics = {
        "epoch/gpu_util_percent": gpu_util,
        "epoch/gpu_mem_used_mb": mem_used_mb,
        "epoch/gpu_mem_total_mb": mem_total_mb,
    }
    if mem_total_mb > 0:
        metrics["epoch/gpu_mem_used_percent"] = (mem_used_mb / mem_total_mb) * 100.0
    return metrics


class DummyProgressBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass


class DistributedTqdmProgressBar:
    def __init__(
        self, accelerator, epochs: int, num_steps_per_epoch: int | None, **kwargs
    ) -> None:
        self.accelerator = accelerator
        self.epochs = epochs
        self.current_epoch = 1
        self.num_steps_per_epoch = num_steps_per_epoch
        self.tqdm_kwargs = kwargs

    def on_epoch_start(self):
        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(total=self.num_steps_per_epoch, **self.tqdm_kwargs)
        else:
            self.progress_bar = DummyProgressBar()

    def update(self, n: int = 1) -> None:
        self.progress_bar.update(n)

    def close(self) -> None:
        self.progress_bar.close()

    def on_epoch_end(self) -> None:
        self.current_epoch += 1
        self.progress_bar.close()

    def show_metrics(self, metrics: dict[str, float]) -> None:
        description = f"Epoch {self.current_epoch}/{self.epochs}"
        for name, score in metrics.items():
            description += f" - {name}: {score:.6f}"
        self.progress_bar.set_description(description)


class LossTracker:
    def __init__(
        self,
        ndigits=4,
    ) -> None:
        self.ndigits = ndigits
        self._loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[float] = []

    def update(self, loss_tensor: torch.Tensor):
        loss = loss_tensor.item()
        self._loss = (self._loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1

    def reset(self):
        self._loss = 0
        self.loss_count = 0

    def on_epoch_end(self, reset: bool = True):
        self.history.append(self.loss)
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        return round(float(self._loss), self.ndigits)
