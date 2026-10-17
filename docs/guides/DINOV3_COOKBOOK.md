---
type: Guide
title: DINOv3 cookbook
description: Getting, verifying and running Meta's DINOv3 backbones locally for photo similarity, with licence terms, links and GPU sizing.
resource: guides/DINOV3_COOKBOOK.md
tags: [docs, guide, dinov3, embeddings, licence]
timestamp: 2026-10-16T00:00:00Z
okf_version: 0.1
---

# DINOv3 cookbook

DINOv3 is Meta AI's self-supervised vision backbone family. We use it as a frozen feature
extractor: a global embedding per image plus a dense grid of patch features, for photo
similarity and near-duplicate checks. This guide covers access, licence, local layout and
working recipes. The weights themselves are never committed.

## Licence: read this first

DINOv3 is under the custom **DINOv3 License**, not an open-source licence:
<https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>.

| Activity | Allowed |
|---|---|
| Download and run locally; embeddings, similarity, near-duplicate detection, culling | Yes |
| Fine-tune on our photos | Yes |
| Commercial product | Yes, subject to the terms |
| Ship weights or derivatives (for example inside an Electron app) | Only with the licence text included and a prominent **"Built with DINOv3"** |
| Publish a benchmark or research result | Must acknowledge DINOv3 |
| Reverse engineer undisclosed components | No |
| Military, nuclear, espionage or weapons use; sanctions or export violations | No |

Meta can change or terminate the licence, gives no warranties, and the user indemnifies
Meta. Whether a privately hosted service counts as distribution needs legal review before
any commercial deployment. Private experiments need nothing extra.

**This repo's rule:** weights stay in `models/dinov3/` (gitignored). Git tracks only this
guide and [`models/dinov3.manifest.json`](../../models/dinov3.manifest.json), which has
file names, sizes, full SHA-256 values and links.

## Get access

There are two routes. Each is gated and approved by hand.

1. **Meta direct download (what we used).** Request at
   <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>. The approval
   email has signed links for every file. They expire in about two days and act as
   credentials: download right away and never save the links.
2. **Hugging Face (`transformers` format, `model.safetensors`).** Collection:
   <https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009>. **Each
   repo is gated separately**, so access to the 7B repo does not cover ViT-B. Loading
   needs a read token in `HF_TOKEN`; never paste it into chats or commit it.

| Model | Params | Hugging Face repo | Weights | Fits 8 GB GPU |
|---|---|---|---|---|
| ViT-S/16 | 21M | [dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | 86 MB | yes |
| ViT-S+/16 | 29M | [dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) | 115 MB | yes |
| ViT-B/16 | 86M | [dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | 343 MB | yes |
| ViT-L/16 | 300M | [dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | 1.2 GB | yes |
| ViT-H+/16 | 840M | [dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | 3.4 GB | yes, small batches |
| ViT-7B/16 | 6.7B | [dinov3-vit7b16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) | 30.9 GB | **no** (about 14 GB in fp16) |
| ViT-L/16, satellite | 300M | [dinov3-vitl16-pretrain-sat493m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) | 1.2 GB | yes (aerial imagery, not birds) |
| ConvNeXt T/S/B/L | 29–198M | [convnext-tiny](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) · [small](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m) · [base](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m) · [large](https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m) | 0.1–0.8 GB | yes |

LVD-1689M means pretrained on 1.7 billion web images; SAT-493M means satellite images. For
photographs use the LVD models.

Other references: code <https://github.com/facebookresearch/dinov3>, project page
<https://ai.meta.com/dinov3/>, and the `MODEL_CARD.md` and `notebooks/` in the code repo.

## Local layout and verification

```text
models/dinov3/                         gitignored: *.pth weights plus SOURCE.md (provenance)
models/dinov3.manifest.json            tracked: names, bytes, sha256, hub entry, HF repo
```

Each file name ends in the first 8 hex digits of its SHA-256. Verify the full hash
against the manifest after every download:

```python
import hashlib, json
from pathlib import Path

manifest = json.loads(Path("models/dinov3.manifest.json").read_text())
for f in manifest["files"]:
    h = hashlib.sha256()
    with open(Path("models/dinov3") / f["file"], "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    print("OK" if h.hexdigest() == f["sha256"] else "MISMATCH", f["file"])
```

Large downloads can drop mid-transfer (we saw a TLS reset on ViT-L). Resume with
`curl -C - --retry 3` and re-check the hash; never use a file that fails.

## Load a backbone

### Route A: official code with the `.pth` files

Clone <https://github.com/facebookresearch/dinov3> and pin a commit (tested:
`6876159a11b4df116f30f667f8c9888617df0751`). **Do not go through `torch.hub` /
`hubconf.py`**: it imports the segmentation and detection heads, which need extra packages
such as `torchmetrics`. Import the backbone builders directly:

```python
import sys
import torch

sys.path.insert(0, "path/to/dinov3")          # the cloned repo
from dinov3.hub.backbones import dinov3_vitb16

model = dinov3_vitb16(pretrained=True,
                      weights="models/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
model = model.eval().cuda()
```

`.pth` files are Python pickles. Load only files whose hash matches the manifest.

### Route B: Hugging Face `transformers`

Needs repo access and `HF_TOKEN`. `transformers` 4.56 or newer (we run 5.3.0) includes the
model:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m").eval().cuda()
out = model(pixel_values=x)                   # out.last_hidden_state: [CLS, 4 registers, patches]
```

`safetensors` files cannot run code when loaded, which makes this the safer format when
access is available.

## Input and outputs

- RGB, normalised with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- Patch size 16: use sides that are multiples of 16. At 336 px you get a 21×21 grid (441
  patches). We letterbox crops to a square with a neutral grey pad so no detail is invented.
- `model.forward_features(x)` (route A) returns:

| Key | Shape | Use |
|---|---|---|
| `x_norm_clstoken` | (N, D) | Global embedding; cosine similarity between images |
| `x_norm_patchtokens` | (N, 441, D) at 336 px | Dense local features; patch matching and localisation |
| `x_storage_tokens` | (N, 4, D) | Register tokens; ignore for similarity |

D is 384 (S), 768 (B), 1024 (L) or 1280 (H+).

## Recipes

### Global similarity

```python
import torch.nn.functional as F

with torch.inference_mode():
    f = model.forward_features(x)
emb = F.normalize(f["x_norm_clstoken"].float(), dim=-1)
distance = 1 - emb[0] @ emb[1]                # 0 = identical
```

### Patch-level difference (what the near-duplicate study uses)

For two crops, match each patch to its most similar patch in the other crop and average
the dissimilarity. This responds to local changes such as a head turn, which a global
embedding can miss.

```python
p = F.normalize(f["x_norm_patchtokens"].float(), dim=-1)    # (2, P, D)
sim = p[0] @ p[1].T
local = torch.cat([1 - sim.max(1).values, 1 - sim.max(0).values])
patch_mean, patch_p95 = local.mean(), local.quantile(0.95)
```

Calibrate every threshold on the model itself. Measure a noise floor from jittered copies
of the same image (small crop shift, JPEG re-encode, brightness change) and never transfer
thresholds from DINOv2, CLIP or MobileNet.

### Throughput and memory on an 8 GB card

- Use `torch.inference_mode()`, batch 16–32 for ViT-S/B, 8–16 for ViT-L and 2–4 for ViT-H+.
- Store patch features as fp16. For 3,000 crops at 336 px, ViT-B patches take about
  2.5 GB per variant.
- Free the GPU between models (`del model; torch.cuda.empty_cache()`).

## Where it is used

- Near-duplicate signal study: `image-scoring-skills/scripts/neardupe_study/signals.py`
  (`dinov3_vitb16`, `dinov3_vitl16`, alongside DINOv2, SSCD and LightGlue). Results go to
  image-scoring-pipeline issue #606.
- Any published comparison must acknowledge DINOv3, as the licence requires.
