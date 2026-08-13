#!/usr/bin/env python3
import json
import torch
from train_gpt import get_args, get_config, make_model


def _calc_parcellation_dim(config):
    if config["use_encoder"]:
        return (
            (
                config["chunk_len"]
                - config["filter_time_length"]
                + 1
                - config["pool_time_length"]
            )
            // config["stride_avg_pool"]
            + 1
        ) * config["n_filters_time"]
    return config["chunk_len"] * 22


def _count_trainable(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def _count_trainable_named(module):
    if module is None:
        return {}
    return {
        name: p.numel()
        for name, p in module.named_parameters(recurse=True)
        if p.requires_grad
    }


def _has_nonzero_grad(module):
    if module is None:
        return False
    for p in module.parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0:
            return True
    return False

def _sum_params(params_dict):
    return sum(params_dict.values())

def _encoder_detailed_params(encoder, num_chunks):
    if encoder is None:
        return {}

    details = {}

    # PatchEmbedding: shallownet and projection
    pe = encoder.patch_embedding
    details["patch_embedding.conv_time"] = _count_trainable(pe.shallownet[0])
    details["patch_embedding.conv_spatial"] = _count_trainable(pe.shallownet[1])
    details["patch_embedding.batchnorm"] = _count_trainable(pe.shallownet[2])
    details["patch_embedding.projection_conv"] = _count_trainable(pe.projection[0])

    # Transformer blocks
    details["transformer.total"] = _count_trainable(encoder.transformer)
    for idx, block in enumerate(encoder.transformer):
        # Structure: block[0] ResidualAdd(attn), block[1] ResidualAdd(ff)
        try:
            attn_ln = block[0].fn[0]
            attn = block[0].fn[1]
            ff_ln = block[1].fn[0]
            ff = block[1].fn[1]
            details[f"transformer.block{idx}.attn.layernorm"] = _count_trainable(attn_ln)
            details[f"transformer.block{idx}.attn.qkv_proj"] = (
                _count_trainable(attn.queries)
                + _count_trainable(attn.keys)
                + _count_trainable(attn.values)
            )
            details[f"transformer.block{idx}.attn.out_proj"] = _count_trainable(attn.projection)
            details[f"transformer.block{idx}.ff.layernorm"] = _count_trainable(ff_ln)
            # FeedForwardBlock is Sequential(Linear, GELU, Dropout, Linear)
            details[f"transformer.block{idx}.ff.linear1"] = _count_trainable(ff[0])
            details[f"transformer.block{idx}.ff.linear2"] = _count_trainable(ff[3])
        except Exception:
            # Fallback to total if structure differs
            details[f"transformer.block{idx}.total"] = _count_trainable(block)

    # Classification head (encoder)
    if hasattr(encoder, "fc") and hasattr(encoder, "final_layer"):
        if hasattr(encoder.fc, "linear_layers"):
            chunk_key = str(num_chunks)
            if chunk_key in encoder.fc.linear_layers:
                details["cls_head.fc_linear"] = _count_trainable(
                    encoder.fc.linear_layers[chunk_key]
                )
        if hasattr(encoder.fc, "fc"):
            # Sequential(ELU, Dropout, Linear, ELU, Dropout)
            details["cls_head.fc_hidden"] = _count_trainable(encoder.fc.fc[2])
        # Final layer linear
        if hasattr(encoder.final_layer, "final_layer"):
            details["cls_head.final_linear"] = _count_trainable(
                encoder.final_layer.final_layer[0]
            )

    details["encoder_total"] = _count_trainable(encoder)
    return details

def main():
    args = get_args().parse_args()
    config = get_config(args)

    derived = {
        "training_style": config["training_style"],
        "is_decoding_mode": config["training_style"] == "decoding",
        "ft_only_encoder": config["ft_only_encoder"],
        "encoder_is_decoding_mode": config["ft_only_encoder"],
        "decoder_is_decoding_mode": config["training_style"] == "decoding",
        "use_encoder": config["use_encoder"],
        "freeze_encoder": config["freeze_encoder"],
        "freeze_decoder": config["freeze_decoder"],
        "freeze_embedder": config["freeze_embedder"],
        "freeze_unembedder": config["freeze_unembedder"],
        "cls_head_layer": config["cls_head_layer"],
        "num_decoding_classes": config["num_decoding_classes"],
        "num_chunks": config["num_chunks"],
        "chunk_len": config["chunk_len"],
        "chunk_ovlp": config["chunk_ovlp"],
        "embedding_dim": config["embedding_dim"],
        "num_encoder_layers": config["num_encoder_layers"],
        "num_hidden_layers": config["num_hidden_layers"],
        "parcellation_dim": _calc_parcellation_dim(config),
        "dummy_input_shape": [2, config["num_chunks"], 22, config["chunk_len"]],
    }

    print("=== Parsed config (selected) ===")
    print(json.dumps(derived, indent=2, ensure_ascii=True))

    # Build model to inspect actual usage
    model = make_model(config)
    model.train()

    # Track whether decoder forward is called
    decoder_called = {"flag": False}

    def _decoder_hook(*_args, **_kwargs):
        decoder_called["flag"] = True

    decoder_hook = model.decoder.register_forward_hook(
        lambda *_args, **_kwargs: _decoder_hook()
    )

    # Build a dummy batch
    batch_size = 2
    num_chunks = config["num_chunks"]
    chunk_len = config["chunk_len"]
    inputs = torch.randn(batch_size * num_chunks, 22, chunk_len)
    attention_mask = torch.ones(batch_size * num_chunks, dtype=torch.long)
    batch = {"inputs": inputs, "attention_mask": attention_mask}

    outputs = model.forward(batch=batch)
    if isinstance(outputs, dict) and "decoding_logits" in outputs:
        loss = outputs["decoding_logits"].sum()
    elif isinstance(outputs, dict) and "outputs" in outputs:
        loss = outputs["outputs"].sum()
    else:
        loss = outputs.sum()

    model.zero_grad(set_to_none=True)
    loss.backward()

    decoder_hook.remove()

    # Parameter stats
    stats = {
        "trainable_params": {
            "encoder": _count_trainable(model.encoder),
            "embedder": _count_trainable(model.embedder),
            "decoder": _count_trainable(model.decoder),
            "unembedder": _count_trainable(model.unembedder),
            "total": _count_trainable(model),
        },
        "forward_usage": {
            "decoder_forward_called": decoder_called["flag"],
        },
        "gradients_updated": {
            "encoder_has_grad": _has_nonzero_grad(model.encoder),
            "embedder_has_grad": _has_nonzero_grad(model.embedder),
            "decoder_has_grad": _has_nonzero_grad(model.decoder),
            "unembedder_has_grad": _has_nonzero_grad(model.unembedder),
        },
        "encoder_detailed_params": _encoder_detailed_params(
            model.encoder, config["num_chunks"]
        ),
    }

    print("=== Runtime checks (dummy forward/backward) ===")
    print(json.dumps(stats, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
