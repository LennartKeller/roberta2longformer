# roberta2longformer

This repo contains various functions to convert the encoder part of a pretrained `RoBerta`-model to long-sequence Transformers.

The memory consumption and runtime of language models with vanilla self-attention grow quadratically to the length of the input sequence.
Various models were proposed to relax this issue either by using sparse or local attention patterns or approximating the full self-attention matrix with decomposition methods.
This repository contains some functions to initialize some of these models with the weights from a pretrained `RoBerta` checkpoint and, therefore, can be helpful to create new models for long-document tasks efficiently.

Note that initializing these models with pretrained weights doesn't make them directly usable, let alone competitive.
In most cases, at least a few thousand "continued-pretraining" steps are required to achieve satisfactory results on any downstream task.
## Roberta

Convert pretrained `RoBerta` models to `Longformer` models

The `Longformer` model ([Beltagy, I., Peters, M. E., & Cohan, A. (2020).](https://arxiv.org/abs/2004.05150)) replaces the full-attention mechanism with local attention patterns and task-specific global attention.
Apart from that, `Longformer` models use the `RoBerta` ([Liu, Y., Ott, M., Goyal, et. al (2019).](https://arxiv.org/abs/1907.11692)) architecture. So it is easily possible to load the weights of a pretrained `RoBerta` model into a `Longformer`.

```python
from roberta2longformer import convert_roberta_to_longformer

from transformers import RobertaModel, RobertaTokenizerFast
from transformers import LongformerModel, LongformerTokenizerFast

roberta_model = RobertaModel.from_pretrained("uklfr/gottbert-base")
roberta_tokenizer = RobertaTokenizerFast.from_pretrained("uklfr/gottbert-base")

longformer_model, longformer_tokenizer = convert_roberta_to_longformer(
    roberta_model=roberta_model,
    roberta_tokenizer=roberta_tokenizer,
    longformer_max_length=8192
)

print(list(longformer_model.encoder.state_dict().items())[0])
print(list(roberta_model.encoder.state_dict().items())[0])

inputs = longformer_tokenizer("Er sah eine irdische Zentralregierung, und er erblickte Frieden, Wohlstand und galaktische Anerkennung."
                              "Es war eine Vision, doch er nahm sie mit vollen Sinnen in sich auf."
                              "Im Laderaum der STARDUST begann eine r??tselhafte Maschine zu summen."
                              "Die dritte Macht nahm die Arbeit auf."
                              "Da l??chelte Perry Rhodan zum blauen Himmel empor."
                              "Langsam l??ste er die Rangabzeichen von dem Schulterst??ck seiner Kombination.",
                              return_tensors="pt")
outputs = longformer_model(**inputs)

# Or to finetune the model on a task:
from transformers import LongformerForSequenceClassification

longformer_model.save_pretrained("tmp/longformer-gottbert")
longformer_tokenizer.save_pretrained("tmp/longformer-gottbert")

seqclass_model = LongformerForSequenceClassification.from_pretrained("tmp/longformer-gottbert/")
...
```

## Nystr??mformer

The `Nystr??mformer`-architecture ([Xiong et. al (2021)](https://arxiv.org/pdf/2102.03902.pdf)) approximates the self-attention mechanism using the Nystr??m matrix-decomposition.
Thus there is no need for dealing with special attention patterns, making these models, in theory, applicable to a wider variety of tasks.
Compared to `Longformer` models, `Nystr??mformers` seem to consume more memory.

```python
from roberta2nystr??mformer import convert_roberta_to_nystromformer

from transformers import RobertaModel, RobertaTokenizerFast
from transformers import NystromformerTokenizerFast, NystromformerModel

roberta_model = RobertaModel.from_pretrained("uklfr/gottbert-base")
roberta_tokenizer = RobertaTokenizerFast.from_pretrained("uklfr/gottbert-base")

nystromformer_model, nystromformer_tokenizer = convert_roberta_to_nystromformer(
    roberta_model=roberta_model,
    roberta_tokenizer=roberta_tokenizer,
    nystromformer_max_length=8192
)

print(list(nystromformer_model.encoder.state_dict().items())[0])
print(list(roberta_model.encoder.state_dict().items())[0])

inputs = nystromformer_tokenizer("Er sah eine irdische Zentralregierung, und er erblickte Frieden, Wohlstand und galaktische Anerkennung."
                                 "Es war eine Vision, doch er nahm sie mit vollen Sinnen in sich auf."
                                 "Im Laderaum der STARDUST begann eine r??tselhafte Maschine zu summen."
                                 "Die dritte Macht nahm die Arbeit auf."
                                 "Da l??chelte Perry Rhodan zum blauen Himmel empor."
                                 "Langsam l??ste er die Rangabzeichen von dem Schulterst??ck seiner Kombination.",
                                 return_tensors="pt")
outputs = nystromformer_model(**inputs)

# Or to finetune the model on a task:
from transformers import NystromformerForSequenceClassification

nystromformer_model.save_pretrained("tmp/nystromformer-gottbert")
nystromformer_model.save_pretrained("tmp/nystromformer-gottbert")

seqclass_model =  NystromformerForSequenceClassification.from_pretrained("tmp/nystromformer-gottbert/")
...

