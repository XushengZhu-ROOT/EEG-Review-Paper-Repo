import os
import sys

import types
from importlib.machinery import ModuleSpec
import torch


def _safe_import_seed_module():
    # Provide a minimal sklearn stub to avoid dependency on sklearn.
    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        metrics_stub = types.ModuleType("sklearn.metrics")
        sklearn_stub.__spec__ = ModuleSpec(name="sklearn", loader=None)
        metrics_stub.__spec__ = ModuleSpec(name="sklearn.metrics", loader=None)

        def _noop(*args, **kwargs):
            raise RuntimeError("sklearn is not available in this environment")

        metrics_stub.confusion_matrix = _noop
        metrics_stub.accuracy_score = _noop
        metrics_stub.balanced_accuracy_score = _noop
        metrics_stub.cohen_kappa_score = _noop
        metrics_stub.f1_score = _noop
        sklearn_stub.metrics = metrics_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.metrics"] = metrics_stub

    # Provide a minimal utils_eval stub to avoid heavy dependencies.
    if "utils_eval" not in sys.modules:
        utils_eval_stub = types.ModuleType("utils_eval")
        utils_eval_stub.__spec__ = ModuleSpec(name="utils_eval", loader=None)

        def _get_metrics_stub(*args, **kwargs):
            raise RuntimeError("utils_eval.get_metrics is not available in this environment")

        utils_eval_stub.get_metrics = _get_metrics_stub
        sys.modules["utils_eval"] = utils_eval_stub

    # Ensure downstream directory is on sys.path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    import linear_probe_EEGPT_Seed_video as seed

    # Force checkpoint load onto CPU to avoid GPU dependency
    torch_load_orig = torch.load

    def _cpu_load(path, *args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return torch_load_orig(path, *args, **kwargs)

    seed.torch.load = _cpu_load
    return seed


def _count_trainable(module_or_param):
    if isinstance(module_or_param, torch.nn.Parameter):
        return module_or_param.numel() if module_or_param.requires_grad else 0
    return sum(p.numel() for p in module_or_param.parameters() if p.requires_grad)


def _format_int(n):
    return f"{n:,}"


def _build_row(name, linear_count, finetune_count):
    return f"| {name} | {linear_count} | {finetune_count} |"


def main():
    seed = _safe_import_seed_module()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "eegpt_mcae_58chs_4s_large4E.ckpt")

    # Linear probe: freeze encoder
    linear_model = seed.LitEEGPTCausal(
        load_path=ckpt_path,
        freeze_encoder=True,
        encoder_lr_ratio=seed.encoder_lr_ratio,
    )
    # Backbone parts for detailed breakdown
    def backbone_parts(model):
        enc = model.target_encoder
        parts = {
            "Backbone - Patch embedding": _count_trainable(enc.patch_embed),
            "Backbone - Channel embedding": _count_trainable(enc.chan_embed),
            "Backbone - Summary token": _count_trainable(enc.summary_token),
            "Backbone - Transformer blocks": _count_trainable(enc.blocks),
            "Backbone - Norm": _count_trainable(enc.norm),
        }
        return parts

    linear_backbone_parts = backbone_parts(linear_model)
    linear_head_parts = {
        "Head - Channel conv (chan_conv)": _count_trainable(linear_model.chan_conv),
        "Head - Classifier (MLP head)": _count_trainable(linear_model.classifier),
    }
    linear_backbone = sum(linear_backbone_parts.values())
    linear_head = sum(linear_head_parts.values())
    linear_total = linear_backbone + linear_head

    # All finetune: train encoder + head
    finetune_model = seed.LitEEGPTCausal(
        load_path=ckpt_path,
        freeze_encoder=False,
        encoder_lr_ratio=seed.encoder_lr_ratio,
    )
    finetune_backbone_parts = backbone_parts(finetune_model)
    finetune_head_parts = {
        "Head - Channel conv (chan_conv)": _count_trainable(finetune_model.chan_conv),
        "Head - Classifier (MLP head)": _count_trainable(finetune_model.classifier),
    }
    finetune_backbone = sum(finetune_backbone_parts.values())
    finetune_head = sum(finetune_head_parts.values())
    finetune_total = finetune_backbone + finetune_head

    rows = []
    for key in [
        "Backbone - Patch embedding",
        "Backbone - Channel embedding",
        "Backbone - Summary token",
        "Backbone - Transformer blocks",
        "Backbone - Norm",
    ]:
        rows.append(
            _build_row(
                key,
                _format_int(linear_backbone_parts[key]),
                _format_int(finetune_backbone_parts[key]),
            )
        )

    for key in [
        "Head - Channel conv (chan_conv)",
        "Head - Classifier (MLP head)",
    ]:
        rows.append(
            _build_row(
                key,
                _format_int(linear_head_parts[key]),
                _format_int(finetune_head_parts[key]),
            )
        )

    rows.append(_build_row("Total trainable", _format_int(linear_total), _format_int(finetune_total)))

    print("| Module | Linear Probe (freeze_encoder=True) | All Finetune (freeze_encoder=False) |")
    print("| --- | ---: | ---: |")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
