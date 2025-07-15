
# HitsBE (Haar Indexation for Time Series Classification)

After studying the fundamentals of transformer-based models, it becomes clear that adapting them to the time series domain is not a trivial task. One striking fact is the limited number of models specifically adapted for the classification of time series.

A common pattern among these alternatives is that most changes are typically focused on the attention mechanism, leaving aside the data embedding process. These models were explicitly designed to work with natural language, and despite their advances in other fields, their architecture is optimized to leverage their full potential in this domain, something that has been widely demonstrated in recent years. Therefore, before attempting significant modifications to their architecture, it is reasonable to first explore how time series data can be adapted to this type of model.

From this idea arises **HitsBE** (Haar Indexation for Time Series BERT Embeddings), a module that provides a vector representation in the model space for time series, aiming to mimic the embedding process used in transformers for natural language. In other words, it attempts to "translate" time series into the internal language of the model.

When a transformer receives a sentence, it segments it into tokens from a vocabulary and embeds them in a vector space through indexation in a matrix. Positional information is then added, and the sequence of vectors is processed through attention mechanisms. **HitsBE** works similarly: it seeks a representation of the series using tokens from a "vocabulary" and complements the information using additional vectors.

In this case, the additional information is provided through the Haar transform applied to each segment of the time series.


## Repository Structure

- **hitsbe/**: Contains the development of the vocabulary and HitsBE itself.
    - **Models**: Contains HitsBERT, a BERT model with HitsBE implemented.

- **experiments/**: Contains various scripts used during the research and development process.
    - **comparison/**: Final experiments comparing different HitsBERT models and the feature extractor.
    - **data/**: Scripts used to transform datasets from the UCR archive for evaluation.
    - **data_pretraining/**: Scripts for processing data used during pretraining.
    - **pretraining/**: Files used and generated during pretraining, including both execution scripts and outputs.
    - **vocabulary/**: Experimental files used to analyze the vocabulary. Also contains `.vocab` files with defined vocabularies.


## Installation

1. Clone the repository:

```sh
git clone https://github.com/angelolmn/HitsBE.git
cd HitsBE
```

2. Run the container:

### GPU Option

```sh
docker build -t hitsbe .
docker run -v /path_to/HitsBE:/hitsbe --gpus device=0 -ti hitsbe:latest
```

> Replace `/path_to/HitsBE` with the local path where you cloned the repository.
> The `-v` option shares files between host and container.
> `-ti` enables an interactive terminal.


3. Install dependencies:
```sh
docker exec -ti <container_id> bash
cd /hitsbe
poetry install
```


## Usage

### Creating the HitsBERT Model

To create a **HitsBERT** model, you first need to define the **HitsBE** configuration using `HitsbeConfig`, where you specify the basic parameters: the vocabulary to use, the time series length, the embedding dimension, the segment size, and the maximum depth of the Haar transform.

Next, define the `BertConfig` configuration from the **transformers** library, as with any standard BERT model. More details about these parameters can be found in section **7.4 of the thesis PDF**.

Finally, instantiate the **HitsBERT** model combining both configurations.

```python
from hitsbe.models import hitsBERT
from hitsbe import Hitsbe, HitsbeConfig
from transformers import BertConfig

hitsbe_config = HitsbeConfig(
    vocabulary_path=filename, 
    ts_len=ts_len,
    dim_model=dim_model, 
    dim_segment=dim_segment, 
    max_haar_depth=max_haar_depth
)

bert_config = BertConfig(
    ...
)

hb = hitsBERT.HitsBERT(bert_config=bert_config, hitsbe_config=hitsbe_config)
```

### Model Pretraining

If you wish to pretrain the model following the proposed pretraining scheme, continue with:

```python
hbpre = hitsBERT.HitsBERTPretraining(model=hb)
```


### Saving the Model

Once pretrained, the model can be saved to disk using the `save_pretrained` method. This method stores both the weights and the configuration files necessary for future loading.

```python
hbpre.model.save_pretrained("path/to/save", "filename.bin")
```

or

```python
hb.save_pretrained("path/to/save", "filename.bin")
```


### Loading the Model

To reuse a previously saved model, load it directly via `from_pretrained`, specifying the path to the saved weights.

```python
hb = HitsBERT.from_pretrained("path/to/save", "filename.bin")
```

If you want to initialize the model with pretrained weights from `bert-base-uncased` or `bert-large-uncased` for example, you must manually define the `BertConfig` and load the weights via `load_state_dict`. This is useful if you want to leverage a BERT pretrained on natural language as a starting point.

```python
hitsbe_config = HitsbeConfig(
    ...
)

bert = BertModel.from_pretrained("bert-base-uncased")
bert_config = bert.config

hb = hitsBERT.HitsBERT(bert_config=bert_config, hitsbe_config=hitsbe_config)
hb.bert.load_state_dict(bert.state_dict())
```


### Running Scripts

Any script in the project can be executed directly with `poetry`:

```sh
poetry run python "path/to/script"
```

This ensures that all dependencies managed by `poetry` are properly activated during execution.

Some scripts may not work correctly because they were developed for earlier versions of HitsBE, the corresponding experiments have already been conducted.