"""CLIP vision encoder, zero-shot heads, task vectors and the evaluator.

Conventions (identical to FusionBench's HFCLIPClassifier):
  logits = logit_scale * normalize(visual_projection(vision_model(x).pooler_output)) @ W^T
where W are the zero-shot text heads of the *pretrained* CLIP model and visual_projection
is the pretrained projection (FusionBench only fine-tunes the vision tower).
Task vectors are tau_i = theta_i - theta_0 over the vision tower's parameters.
"""
import os
import re

import numpy as np
import torch

from .data import CLIP_MEAN, CLIP_STD, TaskData
from .zeroshot import get_classnames_and_templates

_LAYER_RE = re.compile(r"encoder\.layers\.(\d+)\.")


def set_determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def unit_of(name):
    """Depth unit of a vision-tower parameter: 'pre', block index (int), or 'post'."""
    m = _LAYER_RE.search(name)
    if m:
        return int(m.group(1))
    if "post_layernorm" in name:
        return "post"
    return "pre"  # embeddings, pre_layrnorm


def kind_of(name):
    """'attn' (q, k, v, out projections and layer_norm1), 'mlp' (fc1, fc2, layer_norm2) or 'other'."""
    if "self_attn" in name or "layer_norm1" in name:
        return "attn"
    if ".mlp." in name or "layer_norm2" in name:
        return "mlp"
    return "other"


@torch.no_grad()
def build_heads(base_id, tasks, device):
    """Zero-shot heads from the pretrained CLIP text tower (FusionBench formula)."""
    from transformers import CLIPModel, CLIPProcessor

    clip = CLIPModel.from_pretrained(base_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(base_id)
    heads = {}
    for task in tasks:
        classnames, templates = get_classnames_and_templates(task)
        ws = []
        for c in classnames:
            text = [t(c) for t in templates]
            inp = proc(text=text, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            emb = clip.text_model(**inp)[1]
            emb = clip.text_projection(emb)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            emb = emb.mean(dim=0)
            ws.append(emb / emb.norm(p=2, dim=-1, keepdim=True))
        heads[task] = torch.stack(ws).cpu()
    return {"heads": heads,
            "proj": clip.visual_projection.weight.detach().cpu(),
            "logit_scale": float(clip.logit_scale.exp().item())}


def load_vision(model_id, device):
    from transformers import CLIPVisionModel
    m = CLIPVisionModel.from_pretrained(model_id).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class Assets:
    """Pretrained weights, task vectors and heads, all on `device`."""

    def __init__(self, base_id, ft_pattern, tasks, device, cache_dir, tau_device=None):
        self.device = device
        self.tau_device = tau_device or device
        self.tasks = list(tasks)
        os.makedirs(cache_dir, exist_ok=True)
        head_path = os.path.join(cache_dir, "heads.pt")
        if os.path.exists(head_path):
            h = torch.load(head_path)
        else:
            h = build_heads(base_id, self.tasks, device)
            torch.save(h, head_path)
        self.heads = {t: h["heads"][t].to(device) for t in self.tasks}
        self.proj = h["proj"].to(device)
        self.logit_scale = h["logit_scale"]

        self.model = load_vision(base_id, device)
        self.names = [n for n, _ in self.model.named_parameters()]
        self.theta0 = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.n_layers = self.model.config.num_hidden_layers
        self.tau = {}
        for t in self.tasks:
            ft = load_vision(ft_pattern.format(task=t), "cpu")
            ftp = dict(ft.named_parameters())
            self.tau[t] = {n: (ftp[n].detach().to(self.tau_device) - self.theta0[n].to(self.tau_device))
                           for n in self.names}
            del ft
        self.units = {n: unit_of(n) for n in self.names}
        self.kinds = {n: kind_of(n) for n in self.names}
        self.n_classes = {t: int(self.heads[t].shape[0]) for t in self.tasks}

    def params_with_delta(self, delta, alpha):
        return {n: self.theta0[n] + alpha * delta[n].to(self.device) for n in self.names}

    def expert_params(self, task):
        return {n: self.theta0[n] + self.tau[task][n].to(self.device) for n in self.names}


class Evaluator:
    """Loads a parameter dict into one shared model and evaluates cached images."""

    def __init__(self, assets, data_dir, batch_size=500, gpu_cache_gb=8.0):
        self.a = assets
        self.data_dir = data_dir
        self.bs = batch_size
        self.budget = gpu_cache_gb * 1e9
        self.cache = {}  # (task, split) -> (uint8 tensor on device or cpu, labels np)
        self.used = 0.0
        self.td = {}
        dev = assets.device
        self.mean = torch.tensor(CLIP_MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=dev).view(1, 3, 1, 1)
        self._loaded = None

    def taskdata(self, task):
        if task not in self.td:
            self.td[task] = TaskData(self.data_dir, task)
        return self.td[task]

    def images(self, task, split):
        key = (task, split)
        if key not in self.cache:
            td = self.taskdata(task)
            idx = td.indices(split)
            x = torch.from_numpy(np.ascontiguousarray(td.images[idx]))
            nbytes = x.numel()
            if self.used + nbytes <= self.budget:
                x = x.to(self.a.device)
                self.used += nbytes
            self.cache[key] = (x, td.labels[idx])
        return self.cache[key]

    @torch.no_grad()
    def load(self, params, tag=None):
        """Copy a parameter dict into the model (skipped if `tag` is already loaded)."""
        if tag is not None and tag == self._loaded:
            return
        for n, p in self.a.model.named_parameters():
            p.copy_(params[n])
        self._loaded = tag

    def _prep(self, xb):
        xb = xb.to(self.a.device, non_blocking=True).float().div_(255.0)
        return (xb - self.mean) / self.std

    def _logits_from_pooled(self, pooled, task):
        f = pooled @ self.a.proj.T
        f = f / f.norm(p=2, dim=-1, keepdim=True)
        return self.a.logit_scale * f @ self.a.heads[task].T

    @torch.no_grad()
    def correct(self, task, split):
        """Boolean correctness per image of the currently loaded model."""
        x, y = self.images(task, split)
        out = []
        for s in range(0, len(y), self.bs):
            pooled = self.a.model(pixel_values=self._prep(x[s:s + self.bs])).pooler_output
            out.append(self._logits_from_pooled(pooled, task).argmax(-1).cpu().numpy())
        return np.concatenate(out) == y

    # ---------- activation-level tools (used by the `patch` step) ----------
    @torch.no_grad()
    def hidden_states(self, task, split, s, e):
        """Residual stream after every block for images [s:e]: list of L tensors, + logits."""
        x, _ = self.images(task, split)
        o = self.a.model(pixel_values=self._prep(x[s:e]), output_hidden_states=True)
        return [h.clone() for h in o.hidden_states[1:]], self._logits_from_pooled(o.pooler_output, task)

    @torch.no_grad()
    def logits_with_patch(self, task, split, s, e, layer, replacement):
        """Logits when the output of block `layer` is replaced by `replacement`."""
        x, _ = self.images(task, split)
        blk = self.blocks()[layer]

        def hook(_m, _inp, out):
            return (replacement,) + tuple(out[1:]) if isinstance(out, tuple) else replacement

        h = blk.register_forward_hook(hook)
        try:
            pooled = self.a.model(pixel_values=self._prep(x[s:e])).pooler_output
        finally:
            h.remove()
        return self._logits_from_pooled(pooled, task)

    @torch.no_grad()
    def hidden_all(self, task, split, s, e):
        """All residual states (index 0 = input to block 0, l = output of block l-1), + logits."""
        x, _ = self.images(task, split)
        o = self.a.model(pixel_values=self._prep(x[s:e]), output_hidden_states=True)
        return [h for h in o.hidden_states], self._logits_from_pooled(o.pooler_output, task)

    @torch.no_grad()
    def run_block(self, layer, h):
        """Block `layer` of the currently loaded model applied to state h (transformers 4.x/5.x)."""
        import inspect
        blk = self.blocks()[layer]
        if not hasattr(self, "_n_pos"):
            ps = inspect.signature(blk.forward).parameters.values()
            self._n_pos = len([p for p in ps if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty])
        out = blk(h, *([None] * (self._n_pos - 1)))
        return out[0] if isinstance(out, tuple) else out

    def blocks(self):
        """The encoder blocks (works for transformers 4.x and 5.x module layouts)."""
        for name, mod in self.a.model.named_modules():
            if name.endswith("encoder.layers"):
                return mod
        raise RuntimeError("could not find encoder.layers in the vision model")

    def labels(self, task, split):
        return self.images(task, split)[1]
