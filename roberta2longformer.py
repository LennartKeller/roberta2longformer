import torch
from torch import nn

from collections import OrderedDict
from tempfile import TemporaryDirectory
from transformers import LongformerModel, LongformerConfig, LongformerTokenizerFast


def convert_roberta_to_longformer(
    roberta_model,
    roberta_tokenizer,
    longformer_max_length: int = 4096,
    attention_window: int = 512,
):

    ##################################
    # Create new longformer instance #
    ##################################
    longformer_config = LongformerConfig(
        max_position_embeddings=longformer_max_length + 2,
        attention_window=attention_window,
    )
    longformer_model = LongformerModel(longformer_config)

    ###############################
    # Create longformer tokenizer #
    ###############################

    # Longformer tokenizers are Roberta tokenizers.
    # But to follow the conventions
    # and to avoid confusion we create a
    # longformer tokenizer class with the state of
    # the original tokenizer.
    with TemporaryDirectory() as temp_dir:
        roberta_tokenizer.save_pretrained(temp_dir)
        longformer_tokenizer = LongformerTokenizerFast.from_pretrained(temp_dir)
    longformer_tokenizer.model_max_length = longformer_max_length
    longformer_tokenizer.init_kwargs["model_max_length"] = longformer_max_length

    ######################
    # Copy model weights #
    ######################

    # We only copy the encoder weights and resize the embeddings.
    # Pooler weights are kept untouched.

    # ---------#
    # Encoder  #
    # ---------#
    roberta_parameters = roberta_model.encoder.state_dict()
    longformer_parameters = longformer_model.encoder.state_dict()

    # Load all compatible keys directly and obtain missing keys to handle later
    errors = longformer_model.encoder.load_state_dict(roberta_parameters, strict=False)
    assert not errors.unexpected_keys, "Found unexpected keys"
    missing_keys = errors.missing_keys

    # We expect, the keys to be the weights of the global attention modules and
    # reuse roberta's normal attention weights for those modules.
    for longformer_key in missing_keys:
        # Resolve layer properties
        (
            prefix,
            layer_idx,
            layer_class,
            layer_type,
            target,
            params,
        ) = longformer_key.split(".")
        assert layer_class == "attention" or target.endswith(
            "global"
        ), f"Unexcpected parameters {longformer_key}."
        # Copy the normal weights attention weights to the global attention layers too
        roberta_target_key = ".".join(
            [
                prefix,
                layer_idx,
                layer_class,
                layer_type,
                target.removesuffix("_global"),
                params,
            ]
        )
        roberta_weights = roberta_parameters[roberta_target_key]
        longformer_parameters[longformer_key] = roberta_weights

    # Update the state of the longformer model
    longformer_model.encoder.load_state_dict(longformer_parameters, strict=True)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    longformer_model.resize_token_embeddings(len(roberta_tokenizer))

    roberta_embeddings_parameters = roberta_model.embeddings.state_dict()
    embedding_parameters2copy = []

    for key, item in roberta_embeddings_parameters.items():
        if not "position" in key and not "token_type_embeddings" in key:
            embedding_parameters2copy.append((key, item))

    # 2. Positional embeddings
    # The positional embeddings are repeatedly copied over
    # to longformer to match the new max_seq_length

    roberta_pos_embs = roberta_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][:-2]
    roberta_pos_embs_extra = roberta_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][-2:]

    assert (
        roberta_pos_embs.size(0) < longformer_max_length
    ), "Longformer sequence length has to be longer than roberta original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(longformer_max_length / roberta_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    longformer_pos_embs = roberta_pos_embs.repeat((n_copies, 1))
    n_pos_embs_left = longformer_max_length - longformer_pos_embs.size(0)
    longformer_pos_embs = torch.cat(
        [longformer_pos_embs, roberta_pos_embs[:n_pos_embs_left]], dim=0
    )

    # Add the last extra embeddings.
    longformer_pos_embs = torch.cat(
        [longformer_pos_embs, roberta_pos_embs_extra], dim=0
    )

    embedding_parameters2copy.append(
        ("position_embeddings.weight", longformer_pos_embs)
    )

    # Load the embedding weights into the longformer model
    embedding_parameters2copy = OrderedDict(embedding_parameters2copy)
    longformer_model.embeddings.load_state_dict(embedding_parameters2copy, strict=False)

    return longformer_model, longformer_tokenizer
