import sys
import os

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


def wandb_log(prefix, log_dict, round_n=3):
    log_dict = {f"{prefix}/{k}": v for k, v in log_dict.items()}
    if accelerator.is_local_main_process:
        wandb.log(log_dict)
        log_dict = {k: round(v, round_n) for k, v in log_dict.items()}
        console.log(log_dict)


def train_epoch():
    model.train()
    losses = deque(maxlen=training_args.log_every_n_steps)
    step = 0
    for batch in tqdm(train_dl, desc="Step"):
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
            wandb_log("train", {"loss": torch.mean(losses).item()})
        step += 1


def evaluate():
    model.eval()
    y_true = []
    y_pred = []
    for batch in tqdm(val_dl, desc="Step"):
        y = model(batch["image"])
        y_true.append(batch["target"].cpu().numpy())
        y_pred.append(y.argmax(-1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    wandb_log("val", {"acc": acc, "f1": f1, "precision": precision, "recall": recall})


def main():
    global accelerator, training_args, model_args, train_dl, val_dl, optimizer, scheduler, model

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

    # train
    console.rule("Training")
    for epoch in tqdm(range(training_args.n_epochs), desc="Epoch"):
        train_epoch()
        evaluate()


if __name__ == "__main__":
    main()
