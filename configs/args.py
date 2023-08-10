from dataclasses import dataclass


@dataclass
class TrainingArgs:
    lr: float = 1e-4
    lr_schedule: str = "linear_with_warmup"
    lr_warmup_steps: int = 1000
    checkpoint_path: str = "checkpoints"
    output_path: str = "outputs"
    wandb_name: str = None
    wandb_mode: str = "offline"
    wandb_project: str = None
    wandb_dir: str = "wandb"
    train_split: str = "train"
    val_split: str = "test"
    n_epochs: int = 10
    batch_size: int = 32
    seed: int = 0
    dataset: str = "mnist"
    train_collator: str = "default"
    val_collator: str = "default"


@dataclass
class ModelArgs:
    n_layers: int = 4
    hidden_dim: int = 512
