# Yoyodyne 🪀

[![PyPI
version](https://badge.fury.io/py/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![CircleCI](https://circleci.com/gh/CUNY-CL/yoyodyne/tree/master.svg?style=svg&circle-token=37883deeb03d32c8a7b2aa7c34e5143bf514acdd)](https://circleci.com/gh/CUNY-CL/yoyodyne/tree/master)

Yoyodyne provides neural models for small-vocabulary sequence-to-sequence
generation with and without feature conditioning.

These models are implemented using PyTorch and PyTorch Lightning.

While we provide classic `lstm` and `transformer` models, some of the provided
models are particularly well-suited for problems where the source-target
alignments are roughly monotonic (e.g., `transducer`) and/or where source and
target vocabularies are not disjoint and substrings of the source are copied
into the target (e.g., `pointer_generator_lstm`).

## Philosophy

Yoyodyne is inspired by [FairSeq](https://github.com/facebookresearch/fairseq)
but differs on several key points of design:

-   It is for small-vocabulary sequence-to-sequence generation, and therefore
    includes no affordances for machine translation or language modeling.
    Because of this:
    -   It has no plugin interface and the architectures provided are intended
        to be reasonably exhaustive.
    -   There is little need for data preprocessing; it works with TSV files.
-   It has support for using features to condition decoding, with
    architecture-specific code to handle this feature information.
-   🚧 UNDER CONSTRUCTION 🚧: It has exhaustive test suites.
-   🚧 UNDER CONSTRUCTION 🚧: It has performance benchmarks.
-   🚧 UNDER CONSTRUCTION 🚧: Releases are made regularly.

## Install

First install dependencies:

    pip install -r requirements.txt

Then install:

    python setup.py install

Or:

    python setup.py develop

The latter creates a Python module in your environment that updates as you
update the code. It can then be imported like a regular Python module:

```python
import yoyodyne
```

## Usage

For examples, see [`experiments`](experiments). See
[`train.py`](yoyodyne/train.py) and [`predict.py`](yoyodyne/predict.py) for all
model options.

## Architectures

The user specifies the model using the `--arch` flag (and in some cases
additional flags).

-   `feature_invariant_transformer`: This is a variant of the `transformer`
    which uses a learned embedding to distinguish input symbols from features.
    It may be superior to the vanilla transformer when using features.
-   `lstm`: This is an LSTM encoder-decoder, with the initial hidden state
    treated as a learned parameter. By default, the encoder is connected to the
    decoder by an attention mechanism; one can disable this (with
    `--no-attention`), in which case the last non-padding hidden state of the
    encoder is concatenated with the decoder hidden state.
-   `pointer_generator_lstm`: This is an attentive pointer-generator with an
    LSTM backend. Since this model contains a copy mechanism, it may be superior
    to the `lstm` when the input and output vocabularies overlap significantly.
-   `transducer`: This is a transducer with an LSTM backend. On model creation,
    expectation maximization is used to learn a sequence of edit operations, and
    imitation learning is used to train the model to implement the oracle
    policy, with roll-in controlled by the `--oracle-factor` flag (default: 1).
    Since this model assumes monotonic alignment, it may be superior to
    attentive models when the alignment between input and output is roughly
    monotonic and when input and output vocabularies overlap significantly.
-   `transformer`: This is a transformer encoder-decoder with positional
    encoding and layer normalization. The user may wish to specify the number of
    attention heads (with `--nheads`; default: 4).

For all models, the user may also wish to specify:

-   `--dec-layers` (default: 1): number of decoder layers
-   `--embedding` (default: 128): embedding size
-   `--enc-layers` (default: 1): number of encoder layers
-   `--hidden-size` (default: 256): hidden layer size

By default, the `lstm`, `pointer_generator_lstm`, and `transducer` models use an
LSTM bidirectional encoder. One can disable this with the `--no-bidirectional`
flag.

## Training options

-   `--batch-size` (default: 16)
-   `--beta1` (default: .9): $\beta_1$ hyperparameter for the Adam optimizer
    (`--optimizer adam`)
-   `--beta2` (default: .99): $\beta_2$ hyperparameter for the Adam optimizer
    (`--optimizer adam`)
-   `--dropout` (default: .1): dropout probability
-   `--epochs` (default: 20)
-   `--gradient-clip` (default: 0.0)
-   `--label-smoothing` (default: not enabled)
-   `--learning-rate` (required)
-   `--lr-scheduler` (default: not enabled)
-   `--optimizer` (default: "adadelta")
-   `--patience` (default: not enabled)
-   `--wandb` (default: False): enables [Weights &
    Biases](https://wandb.ai/site) tracking
-   `--warmup-steps` (default: 1): warm-up parameter for a linear warm-up
    followed by inverse square root decay schedule (only valid with
    `--lr-scheduler warmupinvsq`)

## Data format

The default data format is a two-column TSV file in which the first column is
the source string and the second the target string.

    source   target

To enable the use of a feature column, one specifies a (non-zero) argument to
`--features-col`. For instance in the SIGMORPHON 2017 shared task, the first
column is the source (a lemma), the second is the target (the inflection), and
the third contains semi-colon delimited feature strings:

    source   target    feat1;feat2;...

this format is specified by `--features-col 3`.

Alternatively, for the SIGMORPHON 2016 shared task data format:

    source   feat1,feat2,...    target

this format is specified by `--features-col 2 --features-sep , --target-col 3`.
