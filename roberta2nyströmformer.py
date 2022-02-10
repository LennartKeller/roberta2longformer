import torch
from collections import OrderedDict
from tempfile import TemporaryDirectory
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer
from transformers import NystromformerConfig, NystromformerModel

def convert_roberta_to_nystromformer(
    roberta_model,
    roberta_tokenizer,
    nystromformer_max_length: int = 50176
):
    with TemporaryDirectory() as temp_dir:
        roberta_tokenizer.save_pretrained(temp_dir)
        roberta_tokenizer.model_max_length = nystromformer_max_length
        nystromformer_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    nystromformer_config = NystromformerConfig.from_dict(roberta_model.config.to_dict())
    nystromformer_config.max_position_embeddings =  nystromformer_max_length # - 2 (?)
    nystromformer_model = NystromformerModel(nystromformer_config)

    # Copy encoder weights
    nystromformer_model.encoder.load_state_dict(roberta_model.encoder.state_dict(), strict=False)

    # ------------#
    # Embeddings  #
    # ------------#
    # There are two types of embeddings:

    # 1. Token embeddings
    # We can simply copy the token embeddings.

    # We have to resize the token embeddings upfront, to make load_state_dict work.
    nystromformer_model.resize_token_embeddings(len(roberta_tokenizer))

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
        roberta_pos_embs.size(0) < nystromformer_max_length
    ), "Longformer sequence length has to be longer than roberta original sequence length"

    # Figure out how many time we need to copy the original embeddings
    n_copies = round(nystromformer_max_length / roberta_pos_embs.size(0))

    # Copy the embeddings and handle the last missing ones.
    longformer_pos_embs = roberta_pos_embs.repeat((n_copies, 1))
    n_pos_embs_left = nystromformer_max_length - longformer_pos_embs.size(0)
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
    nystromformer_model.embeddings.load_state_dict(embedding_parameters2copy, strict=False)


    return nystromformer_model, nystromformer_tokenizer


