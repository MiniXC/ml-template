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
    n_epochs: int = 5
    n_steps: int = None
    batch_size: int = 32
    seed: int = 0
    dataset: str = "mnist"
    train_collator: str = "default"
    val_collator: str = "default"
    log_every_n_steps: int = 100
    do_full_eval: bool = True
    do_save: bool = True
    save_onnx: bool = False
    eval_only: bool = False
    eval_every_n_steps: int = 1000
    eval_every_n_epochs: int = None
    save_every_n_steps: int = 1000
    do_push_to_hub: bool = False
    hub_repo: str = None


@dataclass
class ModelArgs:
    n_layers: int = 4
    hidden_dim: int = 512
