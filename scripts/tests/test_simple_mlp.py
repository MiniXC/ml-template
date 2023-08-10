import sys

sys.path.append(".")  # add root of project to path

import torch
import onnxruntime as ort

from model.simple_mlp import SimpleMLP
from configs.args import ModelArgs

model = SimpleMLP(ModelArgs())


def test_forward_pass():
    x = torch.randn(1, 1)
    y = model(x)
    assert y.shape == (1, 1)


def test_save_load_model(tmp_path):
    model.save_model(tmp_path / "test")
    model.from_pretrained(tmp_path / "test")
    x = torch.randn(1, 1)
    y = model(x)
    assert y.shape == (1, 1)


def test_onnx(tmp_path):
    model.export_onnx(tmp_path / "test")
    ort_session = ort.InferenceSession(tmp_path / "test" / "model.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: model.dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    assert ort_outs[0].shape == (1, 1)
    regular_outs = model(model.dummy_input)
    assert torch.allclose(regular_outs, torch.tensor(ort_outs[0]), atol=1e-06)
