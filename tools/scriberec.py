# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ScribeRec: a text-line recognizer shaped for ignite-xdna's int8 graph engine, plus its ONNX export.

The engine's rules (Ignition docs/QUANTIZATION-GUIDE.md) decide the shape:
- **Backbone, on the NPU:** dense convolutions only (depthwise is lowered dense and costs Cout times its MACs);
  square kernels of 1 or 3 and square strides of 1 or 2; channels in multiples of 32; ReLU; residual adds.
  Every map has at least 20 rows. The input is 80 rows, and two stride-2 stages leave 20 rows x W/4 columns.
- **Tail, on the CPU:**
  - a learned weighting over the 20 rows (a 1-channel conv over height, softmax across rows) collapses each
    column to one C-vector;
  - a 2-layer BiLSTM;
  - a linear layer to the CTC classes: blank, the characters, space.
  Attention, pooling and the LSTM stay off the NPU, as the rules require.
``--export DIR`` writes ``backbone.onnx`` (1x3x80xW -> Cx20xW/4, static, opset 17) and ``tail.onnx``, from
random weights unless ``--weights`` is given. Run where torch is (resnet_env).
"""
import argparse
from pathlib import Path

import torch
from torch import nn


def conv(cin, cout, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = conv(c, c), nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.b(self.a(x)))


class Backbone(nn.Module):
    def __init__(self, widths=(32, 64, 128), blocks=(1, 2, 3), out=128):
        super().__init__()
        c1, c2, c3 = widths
        layers = [conv(3, c1, 2)] + [Res(c1) for _ in range(blocks[0])]
        layers += [conv(c1, c2, 2)] + [Res(c2) for _ in range(blocks[1])]
        layers += [conv(c2, c3)] + [Res(c3) for _ in range(blocks[2])]
        layers += [nn.Sequential(nn.Conv2d(c3, out, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                         # B x 3 x 80 x W -> B x out x 20 x W/4
        return self.net(x)


class Tail(nn.Module):
    def __init__(self, c=128, hidden=128, classes=438):
        super().__init__()
        self.score = nn.Conv2d(c, 1, 1)
        self.lstm = nn.LSTM(c, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden, classes)

    def forward(self, f):                         # B x C x 20 x T -> B x T x classes (log-probabilities)
        wts = torch.softmax(self.score(f), dim=2)
        seq = (f * wts).sum(2).transpose(1, 2)    # B x T x C
        return torch.log_softmax(self.fc(self.lstm(seq)[0]), -1)


class ScribeRec(nn.Module):
    def __init__(self, classes=438, **kw):
        super().__init__()
        self.backbone = Backbone(**kw)
        self.tail = Tail(self.backbone.net[-1][0].out_channels, classes=classes)

    def forward(self, x):
        return self.tail(self.backbone(x))


def macs(model, h=80, w=320):
    total = 0

    def hook(m, i, o):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += o.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
    hs = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(torch.zeros(1, 3, h, w))
    for x in hs:
        x.remove()
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--weights")
    ap.add_argument("--seed", type=int, help="random init seed, so exports at several widths share weights")
    ap.add_argument("--widths", default="32,64,128", help="channels of the three stages")
    ap.add_argument("--blocks", default="1,2,3", help="residual blocks per stage")
    ap.add_argument("--out-ch", type=int, default=128, help="feature channels handed to the CPU tail")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed)
    m = ScribeRec(widths=tuple(int(v) for v in a.widths.split(",")),
                  blocks=tuple(int(v) for v in a.blocks.split(",")), out=a.out_ch).eval()
    if a.weights:
        m.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    out = Path(a.export)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, 80, a.width)
    torch.onnx.export(m.backbone, x, out / "backbone.onnx", input_names=["x"], output_names=["features"],
                      opset_version=17, dynamo=False)
    f = m.backbone(x)
    torch.onnx.export(m.tail, f, out / "tail.onnx", input_names=["features"], output_names=["logprobs"],
                      opset_version=17, dynamo=False, dynamic_axes={"features": {0: "n", 3: "t"}, "logprobs": {0: "n", 1: "t"}})
    n_bb = sum(p.numel() for p in m.backbone.parameters())
    n_tail = sum(p.numel() for p in m.tail.parameters())
    print(f"backbone {n_bb / 1e6:.2f} M params, {macs(m.backbone, 80, a.width) / 1e6:.0f} M MACs at 80x{a.width}; "
          f"tail {n_tail / 1e6:.2f} M params; features {list(f.shape)} -> {out}")


if __name__ == "__main__":
    main()
