from pathlib import Path
import os

import wandb
from huggingface_hub import Repository, create_repo
import huggingface_hub
from git import Repo
from rich.prompt import Prompt
from rich.console import Console
console = Console()

def wandb_update_config(*args):
    if len(args) == 1 and isinstance(args[0], dict):
        for k, v in args[0].items():
            v = v.__dict__
            v = {f"{k}/{_k}": _v for _k, _v in v.items()}
            wandb.config.update(v)
    for arg in args:
        wandb.config.update(arg)

def wandb_init(wandb_name, wandb_project, wandb_dir, wandb_mode):
    os.environ["WANDB_SILENT"] = "true"
    console.rule("Weights & Biases")
    if wandb_mode == "offline":
        console.print(f'logging in [dark_orange][b]OFFLINE[/b][/dark_orange] mode to [magenta][b]{wandb_dir}[/b][/magenta] directory')
    if wandb_mode == "online":
        wandb_dir = Path(wandb_dir)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_last_input = (wandb_dir / "last_input.txt")
        if wandb_last_input.exists():
            last_project, last_name = open(wandb_last_input, 'r').read().split()
        else:
            last_project, last_name = (None, None)
        if wandb_project is None:
            wandb_project = Prompt.ask("wandb project", default=last_project)
        if wandb_name is None:
            wandb_name = Prompt.ask("wandb name", default=last_name)
        console.print(f'logging in [green][b]ONLINE[/b][/green] mode to [magenta][b]{wandb_project}[/b][/magenta] project')
    console.print(f"run name: [magenta][b]{wandb_name}[/b][/magenta]")
    wandb.init(name=wandb_name, project=wandb_project, dir=wandb_dir, mode=wandb_mode)
    os.environ["WANDB_SILENT"] = "false"

def push_to_hub(repo_name, checkpoint_dir, hub_token, commit_message="update model"):
    try:
        create_repo(repo_name, token=hub_token)
    except huggingface_hub.utils._errors.HfHubHTTPError:
        console.print(f'[magenta]{repo_name}[/magenta] already exists')
    try:
        repo = Repository(checkpoint_dir, clone_from=repo_name, token=hub_token)
    except EnvironmentError:
        repo = Repository(checkpoint_dir, clone_from=repo_name, token=hub_token)
    repo.git_pull()
    git_head_commit_url = repo.push_to_hub(
        commit_message=commit_message, blocking=True, auto_lfs_prune=True
    )
    return git_head_commit_url