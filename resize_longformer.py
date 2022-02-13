import torch
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Tuple
from transformers import LongformerPreTrainedModel, LongformerTokenizerFast, LongformerConfig

def resize_longformer(
    model: LongformerPreTrainedModel,
    tokenizer: LongformerTokenizerFast,
    longformer_max_length: int = 512) -> Tuple[LongformerPreTrainedModel, LongformerTokenizerFast]:

    """
    Resize any longformer model (with task specific head)
    """
    longformer_max_length += 2 # due to longformers origins in roberta...
    ###############################
    # Create longformer tokenizer #
    ###############################
    with TemporaryDirectory() as temp_dir:
        tokenizer.save_pretrained(temp_dir)
        longformer_tokenizer = LongformerTokenizerFast.from_pretrained(temp_dir)
    longformer_tokenizer.model_max_length = longformer_max_length - 2
    longformer_tokenizer.init_kwargs["model_max_length"] = longformer_max_length - 2

    ##################################
    # Create new longformer instance #
    ##################################
    longformer_config = LongformerConfig()
    longformer_config.update(model.config.to_dict())
    longformer_config.max_position_embeddings = longformer_max_length
    longformer_model = model.__class__(longformer_config)

    # We can easily copy all weights except the position embeddings and position ids
    orig_weights = model.state_dict()
    embedding_position_ids = orig_weights.pop("longformer.embeddings.position_ids")
    embedding_position_weights = orig_weights.pop("longformer.embeddings.position_embeddings.weight")
    longformer_model.load_state_dict(orig_weights, strict=False)


    if longformer_max_length < tokenizer.model_max_length:
        # New length is short than orig length
        # Slice new position ids
        new_embedding_position_ids = embedding_position_ids[:, :longformer_max_length]
        # Slice the weights
        new_embedding_position_weights = embedding_position_weights[:longformer_max_length, :]
    else:
        # New length is longer than orig length
        # Create new position_ids
        new_embedding_position_ids = torch.arange(longformer_max_length).unsqueeze(0)
        # Copy weights
        n_copies = longformer_max_length // embedding_position_weights.size(0)
        n_pos_embs_left = longformer_max_length - (n_copies * embedding_position_weights.size(0))
        new_embedding_position_weights = embedding_position_weights.repeat(n_copies, 1)
        new_embedding_position_weights = torch.cat([
            new_embedding_position_weights,
            embedding_position_weights[:n_pos_embs_left]
        ], 0)

    embedding_states = OrderedDict({
        "longformer.embeddings.position_ids": new_embedding_position_ids,
        "longformer.embeddings.position_embeddings.weight": new_embedding_position_weights
    })
    longformer_model.load_state_dict(embedding_states, strict=False)



    return longformer_model, longformer_tokenizer
