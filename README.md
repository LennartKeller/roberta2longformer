# roberta2longformer

Convert to pretrained RoBerta models to Longformer models

The memory consumption and runtime of most attention-based language models grows quadratically to the length of the input sequence.
The `Longformer` model ([Beltagy, I., Peters, M. E., & Cohan, A. (2020).](https://arxiv.org/abs/2004.05150)) relaxes this issue by replacing the plain attention mechanism with sparse attention patterns.
Since `Longformer` models only replace the attention heads and use the `RoBerta` ([Liu, Y., Ott, M., Goyal, et. al (2019).](https://arxiv.org/abs/1907.11692)) architecture otherwise, it is possible to load the weights of a pretrained `RoBerta` model into a `Longformer`.

This repo contains a function which takes the weight of the encoder part of a pretrained `RoBerta` model and then creates a new `Longformer` model containing the weights:

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
                              "Im Laderaum der STARDUST begann eine rätselhafte Maschine zu summen."
                              "Die dritte Macht nahm die Arbeit auf."
                              "Da lächelte Perry Rhodan zum blauen Himmel empor."
                              "Langsam löste er die Rangabzeichen von dem Schulterstück seiner Kombination.",
                              return_tensors="pt")
outputs = longformer_model(**inputs)

# Or to finetune the model on a task:
from transformers import LongformerForSequenceClassification

longformer_model = LongformerModel.from_pretrained("tmp/longformer-gottbert")
seqclass_model = LongformerForSequenceClassification.from_pretrained("tmp/longformer-gottbert/")
```
