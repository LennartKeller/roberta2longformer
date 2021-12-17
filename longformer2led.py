import re
from collections import OrderedDict
from tempfile import TemporaryDirectory


from transformers import LEDModel, LEDConfig, LEDTokenizerFast
from transformers import LongformerModel, LongformerTokenizerFast


def convert_longformer_to_led(
    longformer_model, longformer_tokenizer, decoder_output_max_length: int = 4096
):

    # Create LEDTOkenizer with longformers tokenizer state
    with TemporaryDirectory() as temp_dir:
        longformer_tokenizer.save_pretrained(temp_dir)
        led_tokenizer = LEDTokenizerFast.from_pretrained(temp_dir)

    # instanciate new LED model with hparams of longformer
    longformer_config = longformer_model.config

    longformer_config = longformer_model.config.to_dict()

    # Convert params with other names...
    config_mapping = {
        "hidden_size": "d_model",
        "intermediate_size": "encoder_ffn_dim",
        "max_position_embeddings": "max_encoder_position_embeddings",
    }
    converted_config = {
        config_mapping.get(key, key): value for key, value in longformer_config.items()
    }
    # some params have to be set manually
    converted_config["max_encoder_position_embeddings"] = (
        converted_config["max_encoder_position_embeddings"] - 2
    )
    converted_config["max_decoder_position_embeddings"] = decoder_output_max_length

    converted_config["decoder_ffn_dim"] = converted_config["encoder_ffn_dim"]

    led_config = LEDConfig.from_dict(converted_config)
    # Set the decoder start token id to eos token id
    led_config.decoder_start_token_id = led_config.eos_token_id

    led_model = LEDModel(led_config)

    # Copy weights from longformer to encoder part of LDE
    encoder_parameter_mapping = {
        # Self attention query key value (+ global version)
        r"layer\.(?P<layer_idx>\d+)\.attention\.self\.(?P<param_class>(key|query|value)(_global)?)\.(?P<param_type>(weight|bias))": "layers.{layer_idx}.self_attn.longformer_self_attn.{param_class}.{param_type}",
        # self attention output feed forward
        r"layer\.(?P<layer_idx>\d+)\.attention\.output\.dense\.(?P<param_type>(weight|bias))": "layers.{layer_idx}.self_attn.output.{param_type}",
        # self attention layer norm
        r"layer\.(?P<layer_idx>\d+)\.attention\.output\.LayerNorm\.(?P<param_type>(weight|bias))": "layers.{layer_idx}.self_attn_layer_norm.{param_type}",
        # first attention layer dense layer
        r"layer\.(?P<layer_idx>\d+)\.intermediate\.dense\.(?P<param_type>(weight|bias))": "layers.{layer_idx}.fc1.{param_type}",
        # second attention layer dense layer
        r"layer\.(?P<layer_idx>\d+)\.output\.dense\.(?P<param_type>(weight|bias))": "layers.{layer_idx}.fc2.{param_type}",
        # final layer norm
        r"layer\.(?P<layer_idx>\d+)\.output\.LayerNorm\.(?P<param_type>(weight|bias))": "layers.{layer_idx}.final_layer_norm.{param_type}",
    }

    converted_params = []
    for param_name, param in longformer_model.encoder.state_dict().items():
        for key, target in encoder_parameter_mapping.items():
            if match := re.fullmatch(key, param_name):
                converted_params_name = target.format(**match.groupdict())
                converted_params.append((converted_params_name, param))

    # Load params into LED model
    converted_params = OrderedDict(converted_params)
    led_model.encoder.load_state_dict(converted_params, strict=False)

    # How to handle the embeddings (=> They are shared between encoder and decoder..)
    # Copy to encoder and to decoder! Don't forget the positional embeddings...

    # Embedding weights are simply copied to the shared module
    led_model.shared.load_state_dict(
        longformer_model.embeddings.word_embeddings.state_dict()
    )

    # Since encoder and decoder have different sequence lengths we have to copy the positional embeddings weights manually.
    param_name, longformer_positional_embeddings = next(
        iter(longformer_model.embeddings.position_embeddings.state_dict().items())
    )

    # For the encoder part, we can simple remove the first two embeddings which are legacy special embeddings
    led_encoder_positional_embeddings = OrderedDict(
        [(param_name, longformer_positional_embeddings[2:])]
    )
    led_model.encoder.embed_positions.load_state_dict(led_encoder_positional_embeddings)

    # For the decoder part, we copy the N-first positional embeddings (n= max length of decoder)
    # Of course we discard the first two embeddings...
    led_decoder_positional_embeddings = OrderedDict(
        [
            (
                param_name,
                longformer_positional_embeddings[
                    2 : led_config.max_decoder_position_embeddings + 2
                ],
            )
        ]
    )
    led_model.decoder.embed_positions.load_state_dict(led_decoder_positional_embeddings)

    # How to handle the decoder weights???
    # Like so?:
    # Copy weights from longformer to encoder part of LDE
    decoder_parameter_mapping = {
        # Copy longformer global attention KEY weights and biases to led self attention and encoder attention weights
        r"layer\.(?P<layer_idx>\d+)\.attention\.self\.key_global\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.self_attn.k_proj.{param_type}",
            "layers.{layer_idx}.encoder_attn.k_proj.{param_type}",
        ],
        # Copy longformer global attention VALUE weights and biases to led self attention and encoder attention weights
        r"layer\.(?P<layer_idx>\d+)\.attention\.self\.value_global\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.self_attn.v_proj.{param_type}",
            "layers.{layer_idx}.encoder_attn.v_proj.{param_type}",
        ],
        # Copy longformer global attention QUERY weights and biases to led self attention and encoder attention weights
        r"layer\.(?P<layer_idx>\d+)\.attention\.self\.query_global\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.self_attn.q_proj.{param_type}",
            "layers.{layer_idx}.encoder_attn.q_proj.{param_type}",
        ],
        # copy longformer attentention output to self attention and encoder attention outputs
        r"layer\.(?P<layer_idx>\d+)\.attention\.output\.dense\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.self_attn.out_proj.{param_type}",
            "layers.{layer_idx}.encoder_attn.out_proj.{param_type}",
        ],
        # copy long self attention layer norm to self attention and encoder attention layer norm
        r"layer\.(?P<layer_idx>\d+)\.attention\.output\.LayerNorm\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.self_attn_layer_norm.{param_type}",
            "layers.{layer_idx}.encoder_attn_layer_norm.{param_type}",
        ],
        # first attention layer dense layer
        r"layer\.(?P<layer_idx>\d+)\.intermediate\.dense\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.fc1.{param_type}"
        ],
        # second attention layer dense layer
        r"layer\.(?P<layer_idx>\d+)\.output\.dense\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.fc2.{param_type}"
        ],
        # final layer norm
        r"layer\.(?P<layer_idx>\d+)\.output\.LayerNorm\.(?P<param_type>(weight|bias))": [
            "layers.{layer_idx}.final_layer_norm.{param_type}"
        ],
    }

    converted_params = []
    for param_name, param in longformer_model.encoder.state_dict().items():
        for key, targets in decoder_parameter_mapping.items():
            if match := re.fullmatch(key, param_name):
                for target in targets:
                    converted_params_name = target.format(**match.groupdict())
                    converted_params.append((converted_params_name, param))

    # sort params in order of decoder model (probably not necessary...)
    converted_params = list(
        sorted(
            converted_params,
            key=lambda entry: list(led_model.decoder.state_dict().keys()).index(
                entry[0]
            ),
        )
    )
    # Load params into LED model
    converted_params = OrderedDict(converted_params)
    led_model.decoder.load_state_dict(converted_params, strict=False)

    return led_model, led_tokenizer
