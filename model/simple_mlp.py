from pathlib import Path
from collections import OrderedDict

import yaml
import torch
from torch import nn
from transformers.utils.hub import cached_file

from configs.args import ModelArgs
from scripts.util.remote import push_to_hub

class SimpleMLP(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()

        self.mlp = nn.Sequential(OrderedDict([
            ("layer_in_linear", nn.Linear(1, args.hidden_dim)),
            ("layer_in_gelu", nn.GELU()),
        ]))

        for n in range(args.n_layers):
            self.mlp.add_module(f"layer_{n}_linear", nn.Linear(args.hidden_dim, args.hidden_dim))
            self.mlp.add_module(f"layer_{n}_gelu", nn.GELU())

        self.mlp.add_module("layer_out_linear", nn.Linear(args.hidden_dim, 1))

        self.args = args

    def forward(self, x):
        return self.mlp(x)

    def save_model(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        with open(path / "config.yml", "w") as f:
            f.write(yaml.dump(self.args.__dict__, Dumper=yaml.Dumper))

    def save_and_push_to_hub(self, path):
        self.save_model(path)
        return push_to_hub(path)

    @staticmethod
    def from_pretrained(path_or_hubid):
        path = Path(path_or_hubid)
        if path.exists():
            config_file = path / "config.yml"
            model_file = path / "pytorch_model.bin"
        else:
            config_file = cached_file(path_or_hubid, "config.yml")
            model_file = cached_file(path_or_hubid, "pytorch_model.bin")
        args = yaml.load(open(config_file, "r"), Loader=yaml.Loader)
        args = ModelArgs(**args)
        model = SimpleMLP(args)
        model.load_state_dict(torch.load(model_file))
        return model