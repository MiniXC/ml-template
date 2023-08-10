import sys

sys.path.append(".")  # add root of project to path

# torch & hf
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, HfArgumentParser
from datasets import load_dataset

# plotting, logging & etc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from collections import defaultdict, deque
from pathlib import Path
import yaml
from rich.console import Console
from rich.prompt import Prompt

console = Console()

# local imports
from configs.args import TrainingArgs, ModelArgs
from util.remote import wandb_update_config, wandb_init
from model.simple_mlp import SimpleMLP
from collators import get_collator


def wandb_log(prefix, log_dict, round_n=3, print_log=True):
    log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
    if accelerator.is_local_main_process:
        wandb.log(log_dict, step=global_step)
        if print_log:
            log_dict = {k: round(v, round_n) for k, v in log_dict.items()}
            console.log(log_dict)


def train_epoch(epoch):
    global global_step
    model.train()
    losses = deque(maxlen=training_args.log_every_n_steps)
    step = 0
    console.rule(f"Epoch {epoch}")
    last_loss = None
    for batch in train_dl:
        y = model(batch["image"])
        loss = torch.nn.functional.cross_entropy(y, batch["target"])
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.detach())
        if (
            step > 0
            and step % training_args.log_every_n_steps == 0
            and accelerator.is_local_main_process
        ):
            last_loss = torch.mean(torch.tensor(losses)).item()
            wandb_log("train", {"loss": last_loss}, print_log=False)
        if (
            training_args.do_save
            and global_step > 0
            and global_step % training_args.save_every_n_steps == 0
            and accelerator.is_local_main_process
        ):
            # use wandb run id as checkpoint name
            checkpoint_name = wandb.run.name.strip()
            if len(checkpoint_name) == 0:
                checkpoint_name = wandb.run.id
            checkpoint_path = (
                Path(training_args.checkpoint_path)
                / checkpoint_name
                / f"step_{global_step}"
            )
            model.save_model(checkpoint_path, onnx=training_args.save_onnx)
            if training_args.do_push_to_hub:
                model.save_and_push_to_hub(
                    training_args.hub_repo,
                    checkpoint_path,
                    onnx=training_args.save_onnx,
                )
        if training_args.n_steps is not None and global_step >= training_args.n_steps:
            return
        if (
            training_args.eval_every_n_steps is not None
            and global_step > 0
            and global_step % training_args.eval_every_n_steps == 0
            and accelerator.is_local_main_process
        ):
            if training_args.do_full_eval:
                evaluate()
            else:
                evaluate_loss_only()
            console.rule(f"Epoch {epoch}")
        step += 1
        global_step += 1
        if accelerator.is_local_main_process:
            pbar.update(1)
            if last_loss is not None:
                pbar.set_postfix({"loss": f"{last_loss:.3f}"})


def evaluate():
    model.eval()
    y_true = []
    y_pred = []
    losses = []
    console.rule("Evaluation")
    for batch in val_dl:
        y = model(batch["image"])
        loss = torch.nn.functional.cross_entropy(y, batch["target"])
        losses.append(loss.detach())
        y_true.append(batch["target"].cpu().numpy())
        y_pred.append(y.argmax(-1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    wandb_log("val", {"loss": torch.mean(torch.tensor(losses)).item()})
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro")
    wandb_log("val", {"acc": acc, "f1": f1, "precision": precision, "recall": recall})


def evaluate_loss_only():
    model.eval()
    losses = []
    console.rule("Evaluation")
    for batch in val_dl:
        y = model(batch["image"])
        loss = torch.nn.functional.cross_entropy(y, batch["target"])
        losses.append(loss.detach())
    wandb_log("val", {"loss": torch.mean(torch.tensor(losses)).item()})


def main():
    global accelerator, training_args, model_args, train_dl, val_dl, optimizer, scheduler, model, global_step, pbar

    global_step = 0

    parser = HfArgumentParser([TrainingArgs, ModelArgs])

    accelerator = Accelerator()

    # parse args
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yml"):
        with open(sys.argv[1], "r") as f:
            args_dict = yaml.load(f, Loader=yaml.Loader)
        (training_args, model_args) = parser.parse_dict(args_dict)
    else:
        (training_args, model_args) = parser.parse_args_into_dataclasses()

    # wandb
    wandb_name, wandb_project, wandb_dir, wandb_mode = (
        training_args.wandb_name,
        training_args.wandb_project,
        training_args.wandb_dir,
        training_args.wandb_mode,
    )
    wandb_init(wandb_name, wandb_project, wandb_dir, wandb_mode)
    wandb.run.log_code()

    # log args
    console.rule("Arguments")
    console.print(training_args)
    console.print(model_args)
    wandb_update_config(
        {
            "training": training_args,
            "model": model_args,
        }
    )

    # model
    model = SimpleMLP(model_args)

    train_collator = get_collator(training_args.train_collator)
    val_collator = get_collator(training_args.val_collator)

    # dataset
    console.rule("Dataset")

    console.print(f"[green]dataset[/green]: {training_args.dataset}")
    console.print(f"[green]train_split[/green]: {training_args.train_split}")
    console.print(f"[green]val_split[/green]: {training_args.val_split}")

    train_ds = load_dataset(training_args.dataset, split=training_args.train_split)
    val_ds = load_dataset(training_args.dataset, split=training_args.val_split)

    console.print(f"[green]train[/green]: {len(train_ds)}")
    console.print(f"[green]val[/green]: {len(val_ds)}")

    # dataloader
    train_dl = DataLoader(
        train_ds,
        batch_size=training_args.batch_size,
        shuffle=True,
        collate_fn=train_collator,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=training_args.batch_size,
        shuffle=False,
        collate_fn=val_collator,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.lr)

    # scheduler
    if training_args.lr_schedule == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.lr_warmup_steps,
            num_training_steps=len(train_dl) * training_args.n_epochs,
        )
    else:
        raise NotImplementedError(f"{training_args.lr_schedule} not implemented")

    # accelerator
    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, scheduler
    )

    # train/eval
    if training_args.eval_only:
        console.rule("Evaluation")
        evaluate()
        return

    console.rule("Training")
    if training_args.n_steps is not None and training_args.n_epochs is not None:
        raise ValueError("n_steps and n_epochs are mutually exclusive")
    if training_args.n_steps is not None:
        pbar_total = training_args.n_steps
        training_args.n_epochs = training_args.n_steps // len(train_dl) + 1
    else:
        pbar_total = len(train_dl) * training_args.n_epochs
    if (
        training_args.eval_every_n_epochs is not None
        and training_args.eval_every_n_steps is not None
    ):
        raise ValueError(
            "eval_every_n_epochs and eval_every_n_steps are mutually exclusive"
        )
    if training_args.eval_every_n_epochs is not None:
        training_args.eval_every_n_steps = training_args.eval_every_n_epochs * len(
            train_dl
        )
    pbar = tqdm(total=pbar_total, desc="step")
    for i in range(training_args.n_epochs):
        train_epoch(i)
    if training_args.do_save:
        # use wandb run id as checkpoint name
        checkpoint_name = wandb.run.name.strip()
        if len(checkpoint_name) == 0:
            checkpoint_name = wandb.run.id
        checkpoint_path = (
            Path(training_args.checkpoint_path) / checkpoint_name / "final"
        )
        model.save_model(checkpoint_path, onnx=training_args.save_onnx)
        if training_args.do_push_to_hub:
            model.save_and_push_to_hub(
                training_args.hub_repo, checkpoint_path, onnx=training_args.save_onnx
            )

    # wandb sync reminder
    if accelerator.is_local_main_process and training_args.wandb_mode == "offline":
        console.rule("Weights & Biases")
        console.print(
            f"use \n[magenta]wandb sync {Path(wandb.run.dir).parent}[/magenta]\nto sync offline run"
        )


if __name__ == "__main__":
    main()
