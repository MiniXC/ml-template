import sys

sys.path.append(".")  # add root of project to path

from datasets import load_dataset
from torch.utils.data import DataLoader

from configs.args import ModelArgs, TrainingArgs
from collators import get_collator

default_args = TrainingArgs()

train_dataset = load_dataset(default_args.dataset, split=default_args.train_split)
val_dataset = load_dataset(default_args.dataset, split=default_args.val_split)

train_collator = get_collator(default_args.train_collator)
val_collator = get_collator(default_args.val_collator)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=default_args.batch_size,
    shuffle=True,
    collate_fn=train_collator,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=default_args.batch_size,
    shuffle=False,
    collate_fn=val_collator,
)

IN_SHAPE = (1, 28 * 28)
OUT_SHAPE = (1, 10)


def test_train_dataloader():
    for batch in train_dataloader:
        assert batch["image"].shape == (default_args.batch_size, 28 * 28)
        assert batch["target"].shape == (default_args.batch_size,)
        assert batch["target_onehot"].shape == (default_args.batch_size, 10)
        assert batch["image"].max() <= 1
        assert batch["image"].min() >= 0
        assert batch["target"].max() <= 9
        assert batch["target"].min() >= 0
        break


def test_val_dataloader():
    for batch in val_dataloader:
        assert batch["image"].shape == (default_args.batch_size, 28 * 28)
        assert batch["target"].shape == (default_args.batch_size,)
        assert batch["target_onehot"].shape == (default_args.batch_size, 10)
        assert batch["image"].max() <= 1
        assert batch["image"].min() >= 0
        assert batch["target"].max() <= 9
        assert batch["target"].min() >= 0
        break
