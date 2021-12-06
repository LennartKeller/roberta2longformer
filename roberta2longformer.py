from collections import OrderedDict
from tempfile import TemporaryDirectory


def convert_roberta_to_longformer(
    roberta_model,
    roberta_tokenizer,
    longformer_model,
    longformer_max_length: int = None,
):

    if longformer_max_length is None:
        longformer_max_length = longformer_model.config.max_position_embeddings + 1

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
    # Encoder #
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
        orig_weights = longformer_parameters[longformer_key]
        longformer_parameters[longformer_key] = roberta_weights

    # Update the state of the longformer model
    longformer_model.encoder.load_state_dict(longformer_parameters, strict=True)

    # ------------#
    # Embeddings #
    # ------------#
    # There are two types of embeddings:
    # 1. Token embeddings
    # 2. Positional embeddings
    # But we only need to copy the token embeddings
    # while keeping the positional embeddings fixed.

    roberta_embeddings_parameters = roberta_model.embeddings.state_dict()
    embedding_parameters2copy = []
    # We have to resize the token embeddings upfront, to make load_state_dict work.
    longformer_model.resize_token_embeddings(len(roberta_tokenizer))
    for key, item in roberta_embeddings_parameters.items():
        if not "position" in key:
            embedding_parameters2copy.append((key, item))
    embedding_parameters2copy = OrderedDict(embedding_parameters2copy)

    longformer_model.embeddings.load_state_dict(embedding_parameters2copy, strict=False)

    return longformer_model, longformer_tokenizer
