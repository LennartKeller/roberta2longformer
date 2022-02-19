import torch
from typing import Tuple
from collections import OrderedDict
from tempfile import TemporaryDirectory
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import AutoTokenizer
from transformers import BigBirdConfig, BigBirdForMaskedLM, BigBirdForMaskedLM

def convert_roberta_to_bigbird(
    roberta_model: RobertaForMaskedLM,
    roberta_tokenizer: RobertaTokenizer,
    bigbird_max_length: int = 50176
) -> Tuple[BigBirdForMaskedLM, BigBirdForMaskedLM]:
    """
    Note: In contrast to most other conversion functions, this function copies a model with language modeling head.
    """
    with TemporaryDirectory() as temp_dir:
        roberta_tokenizer.save_pretrained(temp_dir)
        roberta_tokenizer.model_max_length = bigbird_max_length
        bigbird_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    bigbird_config = BigBirdConfig.from_dict(roberta_model.config.to_dict())
    bigbird_config.max_position_embeddings =  bigbird_max_length + 2
    bigbird_model = BigBirdForMaskedLM(bigbird_config)

    # Copy encoder weights
    #bigbird_model.base_model.encoder.load_state_dict(roberta_model.base_model.encoder.state_dict(), strict=False)
    bigbird_model.load_state_dict(roberta_model.state_dict(), strict=False)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    bigbird_model.resize_token_embeddings(len(roberta_tokenizer))

    roberta_embeddings_parameters = roberta_model.base_model.embeddings.state_dict()
    embedding_parameters2copy = []

    for key, item in roberta_embeddings_parameters.items():
        if not "position" in key and not "token_type_embeddings" in key:
            embedding_parameters2copy.append((key, item))

    # 2. Positional embeddings
    # The positional embeddings are repeatedly copied over
    # to longformer to match the new max_seq_length

    roberta_pos_embs = roberta_model.base_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][:-2]
    roberta_pos_embs_extra = roberta_model.base_model.embeddings.state_dict()[
        "position_embeddings.weight"
    ][-2:]

    assert (
        roberta_pos_embs.size(0) <= bigbird_max_length
    ), "Longformer sequence length has to be longer than roberta original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(bigbird_max_length / roberta_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    longformer_pos_embs = roberta_pos_embs.repeat((n_copies, 1))
    n_pos_embs_left = bigbird_max_length - longformer_pos_embs.size(0)
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
    bigbird_model.base_model.embeddings.load_state_dict(embedding_parameters2copy, strict=False)


    return bigbird_model, bigbird_tokenizer



