"""StepSaver and resume_checkpoint loader

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Original code: CheckpointSaver + timm.models._helpers by / Copyright 2020 Ross Wightman +
Simplified and modified by Ferran Soler:

- No recovery
- No checkpoint clean up
- Percentage saver instead of epoch saver

"""

import logging
import os
from typing import Any

import torch
from timm.utils.model import unwrap_model, get_state_dict
from timm.models import clean_state_dict

_logger = logging.getLogger(__name__)


def resume_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    loss_scaler: Any = None,
    log_info: bool = True,
):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            if log_info:
                _logger.info("Restoring model state from checkpoint...")
            state_dict = clean_state_dict(checkpoint["state_dict"])
            model.load_state_dict(state_dict)

            if optimizer is not None and "optimizer" in checkpoint:
                if log_info:
                    _logger.info("Restoring optimizer state from checkpoint...")
                optimizer.load_state_dict(checkpoint["optimizer"])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info("Restoring AMP loss scaler state from checkpoint...")
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            # Load checkpointing vars if exist
            if (
                "dataset_size" in checkpoint
                and "init_datapoints" in checkpoint
                and "current_checkpoint" in checkpoint
                and "max_history" in checkpoint
            ):

                # Load checkpointing vars
                dataset_size = int(checkpoint["dataset_size"])
                init_datapoints = int(checkpoint["init_datapoints"])
                current_checkpoint = int(checkpoint["current_checkpoint"])
                max_history = int(checkpoint["max_history"])
                # Log checkpoint load vars
                if log_info:
                    _logger.info(
                        "Loaded checkpoint '{}': dataset size {}, init datapoints {}, current checkpoint {}, max history {}".format(
                            checkpoint_path,
                            dataset_size,
                            init_datapoints,
                            current_checkpoint,
                            max_history,
                        )
                    )
                return dataset_size, init_datapoints, current_checkpoint, max_history

        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

    return None  # Return None by default


class UpdateSaver:
    def __init__(
        self,
        model,
        optimizer,
        dataset_size,
        global_batch_size,
        total_updates,
        aug_repeats,
        args=None,
        model_ema=None,
        amp_scaler=None,
        save_prefix="ckp/model",
        current_checkpoint=1,
        max_history=10,
        unwrap_fn=unwrap_model,
    ):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.aug_repeats = aug_repeats
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler
        self.current_checkpoint = current_checkpoint

        # config
        self.max_history = max_history
        self.save_prefix = save_prefix
        self.extension = ".pth.tar"
        self.unwrap_fn = unwrap_fn

        # Compute updates to save
        num_updates_save = total_updates // max_history
        self.updates_save = [
            i * num_updates_save for i in range(1, self.max_history + 1)
        ]

    def step_save(self, num_updates):

        if num_updates in self.updates_save:
            # Save and increase current checkpoint idx
            save_idf = (
                "ckp" + str(self.current_checkpoint) + "-" + str(self.max_history)
            )
            self.save(
                save_idf=save_idf,
                datapoints=self.global_batch_size
                / self.aug_repeats
                * num_updates,  # Take repeats into account
            )
            self.current_checkpoint += 1

    def save(self, save_idf, datapoints):
        save_state = {
            "dataset_size": self.dataset_size,
            "init_datapoints": datapoints,
            "current_checkpoint": self.current_checkpoint,
            "max_history": self.max_history,
            "arch": type(self.model).__name__.lower(),
            "state_dict": get_state_dict(self.model, self.unwrap_fn),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.args is not None:
            save_state["arch"] = self.args.model
            save_state["args"] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state["state_dict_ema"] = get_state_dict(
                self.model_ema, self.unwrap_fn
            )
        torch.save(save_state, self.save_prefix + "/" + save_idf + self.extension)
