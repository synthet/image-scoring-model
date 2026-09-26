"""RTMDet-tiny (COCO) as a dependency-free PyTorch module (torch + torchvision only).

Mirrors the public OpenMMLab architecture (CSPNeXt backbone, CSPNeXtPAFPN neck, RTMDetSepBNHead) with
parameter names identical to the upstream checkpoint, so
`rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth` loads with strict=True. Apache-2.0 weights;
this file is our own code.
"""
from __future__ import annotations

import pickle
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

MEAN = (103.53, 116.28, 123.675)  # BGR order, as the upstream data_preprocessor
STD = (57.375, 57.12, 58.395)
STRIDES = (8, 16, 32)


class ConvModule(nn.Module):
    def __init__(self, cin, cout, k, s=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=1e-5, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWSeparable(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.depthwise_conv = ConvModule(cin, cin, k, groups=cin)
        self.pointwise_conv = ConvModule(cin, cout, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    def __init__(self, c, add_identity):
        super().__init__()
        self.conv1 = ConvModule(c, c, 3)
        self.conv2 = DWSeparable(c, c, 5)
        self.add = add_identity

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class ChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Conv2d(c, c, 1, bias=True)

    def forward(self, x):
        return x * F.hardsigmoid(self.fc(F.adaptive_avg_pool2d(x, 1)))


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n, add_identity, attention):
        super().__init__()
        mid = cout // 2
        self.main_conv = ConvModule(cin, mid, 1)
        self.short_conv = ConvModule(cin, mid, 1)
        self.final_conv = ConvModule(2 * mid, cout, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, add_identity) for _ in range(n)])
        self.attention = ChannelAttention(2 * mid) if attention else None

    def forward(self, x):
        y = torch.cat([self.blocks(self.main_conv(x)), self.short_conv(x)], 1)
        if self.attention is not None:
            y = self.attention(y)
        return self.final_conv(y)


class SPPF(nn.Module):
    def __init__(self, cin, cout, ks=(5, 9, 13)):
        super().__init__()
        mid = cin // 2
        self.conv1 = ConvModule(cin, mid, 1)
        self.poolings = nn.ModuleList(nn.MaxPool2d(k, 1, k // 2) for k in ks)
        self.conv2 = ConvModule(mid * (len(ks) + 1), cout, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.poolings], 1))


class CSPNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvModule(3, 12, 3, 2), ConvModule(12, 12, 3), ConvModule(12, 24, 3))
        self.stage1 = nn.Sequential(ConvModule(24, 48, 3, 2), CSPLayer(48, 48, 1, True, True))
        self.stage2 = nn.Sequential(ConvModule(48, 96, 3, 2), CSPLayer(96, 96, 1, True, True))
        self.stage3 = nn.Sequential(ConvModule(96, 192, 3, 2), CSPLayer(192, 192, 1, True, True))
        self.stage4 = nn.Sequential(ConvModule(192, 384, 3, 2), SPPF(384, 384), CSPLayer(384, 384, 1, False, True))

    def forward(self, x):
        x = self.stage1(self.stem(x))
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5


class PAFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_layers = nn.ModuleList([ConvModule(384, 192, 1), ConvModule(192, 96, 1)])
        self.top_down_blocks = nn.ModuleList([CSPLayer(384, 192, 1, False, False), CSPLayer(192, 96, 1, False, False)])
        self.downsamples = nn.ModuleList([ConvModule(96, 96, 3, 2), ConvModule(192, 192, 3, 2)])
        self.bottom_up_blocks = nn.ModuleList([CSPLayer(192, 192, 1, False, False), CSPLayer(384, 384, 1, False, False)])
        self.out_convs = nn.ModuleList([ConvModule(96, 96, 3), ConvModule(192, 96, 3), ConvModule(384, 96, 3)])

    def forward(self, feats):
        inner = [feats[2]]
        for i, idx in enumerate((2, 1)):
            high = self.reduce_layers[i](inner[0])
            inner[0] = high
            up = F.interpolate(high, scale_factor=2, mode="nearest")
            inner.insert(0, self.top_down_blocks[i](torch.cat([up, feats[idx - 1]], 1)))
        outs = [inner[0]]
        for i in range(2):
            outs.append(self.bottom_up_blocks[i](torch.cat([self.downsamples[i](outs[-1]), inner[i + 1]], 1)))
        return [conv(o) for conv, o in zip(self.out_convs, outs)]


class SepBNHead(nn.Module):
    def __init__(self, num_classes=80, c=96, stacked=2):
        super().__init__()
        self.cls_convs = nn.ModuleList(nn.Sequential(*[ConvModule(c, c, 3) for _ in range(stacked)]) for _ in STRIDES)
        self.reg_convs = nn.ModuleList(nn.Sequential(*[ConvModule(c, c, 3) for _ in range(stacked)]) for _ in STRIDES)
        self.rtm_cls = nn.ModuleList(nn.Conv2d(c, num_classes, 1) for _ in STRIDES)
        self.rtm_reg = nn.ModuleList(nn.Conv2d(c, 4, 1) for _ in STRIDES)

    def forward(self, feats):
        scores, boxes = [], []
        for i, (x, s) in enumerate(zip(feats, STRIDES)):
            cls = self.rtm_cls[i](self.cls_convs[i](x))
            dist = self.rtm_reg[i](self.reg_convs[i](x)) * s
            b, _, h, w = cls.shape
            ys, xs = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
            pts = torch.stack([xs, ys], -1).reshape(-1, 2).float() * s
            d = dist.permute(0, 2, 3, 1).reshape(b, -1, 4)
            boxes.append(torch.cat([pts - d[..., :2], pts + d[..., 2:]], -1))
            scores.append(cls.permute(0, 2, 3, 1).reshape(b, -1, cls.shape[1]).sigmoid())
        return torch.cat(boxes, 1), torch.cat(scores, 1)


class RTMDetTiny(nn.Module):
    """Input: BGR float tensor (N,3,640,640) already mean/std-normalized. Output: boxes (N,8400,4) in input
    pixels, scores (N,8400,80)."""

    def __init__(self):
        super().__init__()
        self.backbone = CSPNeXt()
        self.neck = PAFPN()
        self.bbox_head = SepBNHead()

    def forward(self, x):
        return self.bbox_head(self.neck(self.backbone(x)))


def postprocess(boxes, scores, pre_top_k=5000, iou=0.65, score_thr=0.001, per_class=200, keep=300):
    """Batch size 1: top-k by max class score, per-class NMS, then keep the best `keep`."""
    b, s = boxes[0], scores[0]
    top = s.max(1).values.topk(min(pre_top_k, s.shape[0])).indices
    b, s = b[top], s[top]
    cand = (s > score_thr).nonzero()
    bb, ss, cc = b[cand[:, 0]], s[cand[:, 0], cand[:, 1]], cand[:, 1]
    k = batched_nms(bb, ss, cc, iou)
    out_k = []
    per = {}
    for i in k.tolist():
        c = int(cc[i])
        if per.get(c, 0) < per_class:
            per[c] = per.get(c, 0) + 1
            out_k.append(i)
    out_k = torch.tensor(out_k, dtype=torch.long)[:keep] if out_k else torch.zeros(0, dtype=torch.long)
    return bb[out_k], ss[out_k], cc[out_k]


def load_upstream(path: str) -> RTMDetTiny:
    """Load the upstream checkpoint without mmengine installed (its metadata objects are stubbed)."""

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith(("mmengine", "mmdet", "mmcv")):
                return _Stub
            return super().find_class(mod, name)

    pm = types.ModuleType("pm")
    pm.Unpickler, pm.load, pm.__name__ = _Unpickler, pickle.load, "pickle"
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=pm)
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("data_preprocessor")}
    m = RTMDetTiny()
    m.load_state_dict(sd, strict=True)
    return m.eval()


def letterbox(rgb, size=640, pad=114):
    """Uniform-scale, centred letterbox. Returns (BGR normalized tensor 1x3xSxS, scale, padx, pady)."""
    import numpy as np
    from PIL import Image

    h, w = rgb.shape[:2]
    sc = size / max(h, w)
    nw, nh = round(w * sc), round(h * sc)
    im = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    canvas = np.full((size, size, 3), pad, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = im
    x = canvas[..., ::-1].astype(np.float64)
    x = (x - np.array(MEAN)) / np.array(STD)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)), sc, px, py
