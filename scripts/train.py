import sys
import os

sys.path.append(".")  # add root of project to path

# torch & hf
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup, HfArgumentParser
from datasets import load_dataset

# from datasets import load_dataset

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


def epoch(
    dataloader,
    model,
    optimizer,
    scheduler,
):
    global args
    model.train()
    losses = deque()


def evaluate(
    dataloader,
    model,
):
    pass


def main():
    parser = HfArgumentParser([TrainingArgs, ModelArgs])

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
    model.save_model(Path(training_args.checkpoint_path) / "test")

    train_collator = get_collator(training_args.train_collator)
    val_collator = get_collator(training_args.val_collator)

    # dataset
    console.rule("Dataset")
    train_ds = load_dataset(training_args.dataset, split=training_args.train_split)
    val_ds = load_dataset(training_args.dataset, split=training_args.val_split)

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

    for item in train_dl:
        print(item)
        break


if __name__ == "__main__":
    main()
