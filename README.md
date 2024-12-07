# Yoyodyne 🪀

[![PyPI
version](https://badge.fury.io/py/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/CUNY-CL/yoyodyne/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/CUNY-CL/yoyodyne/tree/master)

Yoyodyne provides neural models for small-vocabulary sequence-to-sequence
generation with and without feature conditioning.

These models are implemented using [PyTorch](https://pytorch.org/) and
[Lightning](https://www.pytorchlightning.ai/).

While we provide classic LSTM and transformer models, some of the provided
models are particularly well-suited for problems where the source-target
alignments are roughly monotonic (e.g., `transducer` and `hard_attention_lstm`)
and/or where source and target vocabularies have substantial overlap (e.g.,
`pointer_generator_lstm`).

## Philosophy

Yoyodyne is inspired by [FairSeq](https://github.com/facebookresearch/fairseq)
(Ott et al. 2019) but differs on several key points of design:

-   It is for small-vocabulary sequence-to-sequence generation, and therefore
    includes no affordances for machine translation or language modeling.
    Because of this:
    -   The architectures provided are intended to be reasonably exhaustive.
    -   There is little need for data preprocessing; it works with TSV files.
-   It has support for using features to condition decoding, with
    architecture-specific code for handling feature information.
-   It supports the use of validation accuracy (not loss) for model selection
    and early stopping.
-   Releases are made regularly.
-   🚧 UNDER CONSTRUCTION 🚧: It has exhaustive test suites.
-   🚧 UNDER CONSTRUCTION 🚧: It has performance benchmarks.

## Authors

Yoyodyne was created by [Adam Wiemerslage](https://adamits.github.io/), [Kyle
Gorman](https://wellformedness.com/), Travis Bartley, and [other
contributors](https://github.com/CUNY-CL/yoyodyne/graphs/contributors) like
yourself.

## Installation

### Local installation

Yoyodyne currently supports Python 3.9 through 3.12.

First install dependencies:

    pip install -r requirements.txt

Then install:

    pip install .

It can then be imported like a regular Python module:

```python
import yoyodyne
```

### Google Colab

Yoyodyne is compatible with [Google Colab](https://colab.research.google.com/)
GPU runtimes. [This
notebook](https://colab.research.google.com/drive/1O4VWvpqLrCxxUvyYMbGH9HOyXQSoh5bP?usp=sharing)
provides a worked example. Colab also provides access to TPU runtimes, but this
is not yet compatible with Yoyodyne to our knowledge.

## Usage

### Training

Training is performed by the [`yoyodyne-train`](yoyodyne/train.py) script. One
must specify the following required arguments:

-   `--model_dir`: path for model metadata and checkpoints
-   `--train`: path to TSV file containing training data
-   `--val`: path to TSV file containing validation data

The user can also specify various optional training and architectural arguments.
See below or run [`yoyodyne-train --help`](yoyodyne/train.py) for more
information.

### Validation

Validation is run at intervals requested by the user. See `--val_check_interval`
and `--check_val_every_n_epoch`
[here](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api).
Additional evaluation metrics can also be requested with `--eval_metric`. For
example

    yoyodyne-train --eval_metric ser ...

will additionally compute symbol error rate (SER) each time validation is
performed. Additional metrics can be added to
[`evaluators.py`](yoyodyne/evaluators.py).

### Prediction

Prediction is performed by the [`yoyodyne-predict`](yoyodyne/predict.py) script.
One must specify the following required arguments:

-   `--arch`: architecture, matching the one used for training
-   `--model_dir`: path for model metadata
-   `--checkpoint`: path to checkpoint
-   `--predict`: path to file containing data to be predicted
-   `--output`: path for predictions

The `--predict` file can either be a TSV file or an ordinary TXT file with one
source string per line; in the latter case, specify `--target_col 0`. Run
[`yoyodyne-predict --help`](yoyodyne/predict.py) for more information.

Beam search is implemented (currently only for LSTM-based models) and can be
enabled by setting `--beam_width` \> 1. When using beam search, the
log-likelihood for each hypothesis is always returned. The outputs are pairs of
hypotheses and the associated log-likelihoods.

## Data format

The default data format is a two-column TSV file in which the first column is
the source string and the second the target string.

    source   target

To enable the use of a feature column, one specifies a (non-zero) argument to
`--features_col`. For instance in the [SIGMORPHON 2017 shared
task](https://sigmorphon.github.io/sharedtasks/2017/), the first column is the
source (a lemma), the second is the target (the inflection), and the third
contains semi-colon delimited feature strings:

    source   target    feat1;feat2;...

this format is specified by `--features_col 3`.

Alternatively, for the [SIGMORPHON 2016 shared
task](https://sigmorphon.github.io/sharedtasks/2016/) data:

    source   feat1,feat2,...    target

this format is specified by `--features_col 2 --features_sep , --target_col 3`.

In order to ensure that targets are ignored during prediction, one can specify
`--target_col 0`.

## Reserved symbols

Yoyodyne reserves symbols of the form `<...>` for internal use.
Feature-conditioned models also use `[...]` to avoid clashes between feature
symbols and source and target symbols, and `--no_tie_embeddings` uses `{...}` to
avoid clashes between source and t arget symbols. Therefore, users should not
provide any symbols of the form `<...>`, `[...]`, or `{...}`.

## Model checkpointing

Checkpointing is handled by
[Lightning](https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html).
The path for model information, including checkpoints, is specified by
`--model_dir` such that we build the path `model_dir/version_n`, where each run
of an experiment with the same `model_dir` is namespaced with a new version
number. A version stores all of the following:

-   the index (`model_dir/index.pkl`),
-   the hyperparameters (`model_dir/lightning_logs/version_n/hparams.yaml`),
-   the metrics (`model_dir/lightning_logs/version_n/metrics.csv`), and
-   the checkpoints (`model_dir/lightning_logs/version_n/checkpoints`).

By default, each run initializes a new model from scratch, unless the
`--train_from` argument is specified. To continue training from a specific
checkpoint, the **full path to the checkpoint** should be specified with for the
`--train_from` argument. This creates a new version, but starts training from
the provided model checkpoint.

By default 1 checkpoint is saved. To save more than one checkpoint, use the
`--num_checkpoints` flag. To save a checkpoint every epoch, set
`--num_checkpoints -1`. By default, the checkpoints saved are those which
maximize validation accuracy. To instead select checkpoints which minimize
validation loss, set `--checkpoint_metric loss`.

## Models

The user specifies the overall architecture for the model using the `--arch`
flag. The value of this flag specifies the decoder's architecture and whether or
not an attention mechanism is present. This flag also specifies a default
architecture for the encoder(s), but it is possible to override this with
additional flags. Supported values for `--arch` are:

-   `attentive_gru`: This is an GRU decoder with GRU encoders (by default) and
    an attention mechanism. The initial hidden state is treated as a learned
    parameter.
-   `attentive_lstm`: This is similar to the `attentive_gru` but instead uses an
    LSTM decoder and encoder (by default).
-   `gru`: This is an GRU decoder with GRU encoders (by default); in lieu of an
    attention mechanism, the last non-padding hidden state of the encoder is
    concatenated with the decoder hidden state.
-   `hard_attention_gru`: This is an GRU encoder/decoder modeling generation as
    a Markov process. By default, it assumes a non-monotonic progression over
    the source string, but with `--enforce_monotonic` the model must progress
    over each source character in order. A non-zero value of
    `--attention_context` (default: `0`) widens the context window for
    conditioning state transitions to include one or more previous states.
-   `hard_attention_lstm`: This is similar to the `hard_attention_gru` but
    instead uses an LSTM decoder and encoder (by deafult). `--attention_context`
    (default: `0`) widens the context window for conditioning state transitions
    to include one or more previous states.
-   `lstm`: This is similar to the `gru` but instead uses an LSTM decoder and
    encoder (by default).
-   `pointer_generator_gru`: This is an GRU decoder with GRU encoders (by
    default) and a pointer-generator mechanism. Since this model contains a copy
    mechanism, it may be superior to an ordinary attentive GRU when the source
    and target vocabularies overlap significantly. Note that this model requires
    that the number of `--encoder_layers` and `--decoder_layers` match.
-   `pointer_generator_lstm`: This is similar to the `pointer_generator_gru` but
    instead uses an LSTM decoder and encoder (by default).
-   `pointer_generator_transformer`: This is similar to the
    `pointer_generator_gru` and `pointer_generator_lstm` but instead uses a
    transformer decoder and encoder (by default). When using features, the user
    may wish to specify the number of features attention heads (with
    `--features_attention_heads`).
-   `transducer_gru`: This is an GRU decoder with GRU encoders (by default) and
    a neural transducer mechanism. On model creation, expectation maximization
    is used to learn a sequence of edit operations, and imitation learning is
    used to train the model to implement the oracle policy, with roll-in
    controlled by the `--oracle_factor` flag (default: `1`). Since this model
    assumes monotonic alignment, it may be superior to attentive models when the
    alignment between input and output is roughly monotonic and when input and
    output vocabularies overlap significantly.
-   `transducer_lstm`: This is similar to the `transducer_gru` but instead uses
    an LSTM decoder and encoder (by default).
-   `transformer`: This is a transformer decoder with transformer encoders (by
    default). Sinusodial positional encodings and layer normalization are used.
    The user may wish to specify the number of attention heads (with
    `--source_attention_heads`; default: `4`).

The `--arch` flag specifies the decoder type; the user can override default
encoder types using the `--source_encoder_arch` flag and, when features are
present, the `--features_encoder_arch` flag. Valid values are:

-   `feature_invariant_transformer` (`--source_encoder_arch` only): a variant of
    the transformer encoder used with features; it concatenates source and
    features and uses a learned embedding to distinguish between source and
    features symbols.
-   `linear`: a linear encoder.
-   `gru`: a GRU encoder.
-   `lstm`: a LSTM encoder.
-   `transformer`: a transformer encoder.

For all models, the user may also wish to specify:

-   `--decoder_layers` (default: `1`): number of decoder layers
-   `--embedding` (default: `128`): embedding size
-   `--encoder_layers` (default: `1`): number of encoder layers
-   `--hidden_size` (default: `512`): hidden layer size

By default, RNN-backed (i.e., GRU and LSTM) encoders are bidirectional. One can
disable this with the `--no_bidirectional` flag.

## Training options

A non-exhaustive list includes:

-   Batch size:
    -   `--batch_size` (default: `32`)
    -   `--accumulate_grad_batches` (default: not enabled)
-   Regularization:
    -   `--dropout` (default: `0.2`)
    -   `--label_smoothing` (default: `0.0`)
    -   `--gradient_clip_val` (default: not enabled)
-   Optimizer:
    -   `--learning_rate` (default: `0.001`)
    -   `--optimizer` (default: `"adam"`)
    -   `--beta1` (default: `0.9`): $\beta_1$ hyperparameter for the Adam
        optimizer (`--optimizer adam`)
    -   `--beta2` (default: `0.99`): $\beta_2$ hyperparameter for the Adam
        optimizer (`--optimizer adam`)
    -   `--scheduler` (default: not enabled)
-   Duration:
    -   `--max_epochs`
    -   `--min_epochs`
    -   `--max_steps`
    -   `--min_steps`
    -   `--max_time`
-   Seeding:
    -   `--seed`
-   [Weights & Biases](https://wandb.ai/site):
    -   `--log_wandb` (default: `False`): enables Weights & Biases tracking; the
        "project" name can be specified using the environmental variable
        `$WANDB_PROJECT`.

Additional training options are discussed below.

### Early stopping

To enable early stopping, use the `--patience` and `--patience_metric` flags.
Early stopping occurs after `--patience` epochs with no improvement (when
validation loss stops decreasing if `--patience_metric loss`, or when validation
accuracy stops increasing if `--patience_metric accuracy`). Early stopping is
not enabled by default.

### Schedulers

By default, Yoyodyne uses a constant learning rate during training, but best
practice is to gradually decrease learning rate as the model approaches
convergence using a [scheduler](yoyodyne/schedulers.py). The following
schedulers are supported and are selected with `--scheduler`:

-   `reduceonplateau`: reduces the learning rate (multiplying it by
    `--reduceonplateau_factor`) after `--reduceonplateau_patience` epochs with
    no improvement (when validation loss stops decreasing if
    `--reduceonplateau loss`, or when validation accuracy stops increasing if
    `--reduceonplateau_metric accuracy`) until the learning rate is less than or
    equal to `--min_learning_rate`.
-   `warmupinvsqrt`: linearly increases the learning rate from 0 to
    `--learning_rate` for `--warmup_steps` steps, then decreases learning rate
    according to an inverse root square schedule.

## Tied embeddings

By default, the source and target vocabularies are shared. This can be disabled
with the flag `--no_tie_embeddings`, which uses `{...}` to avoid clashes between
source and target symbols.

### Batch size tricks

**Choosing a good batch size is key to fast training and optimal performance.**
Batch size is specified by the `--batch_size` flag.

One may wish to train with a larger batch size than will fit in "in core". For
example, suppose one wishes to fit with a batch size of 4,096, but this gives an
out of memory (OOM) exception. Then, with minimal overhead, one could simulate
an effective batch size of 4,096 by using batches of size 1,024, [accumulating
gradients from 4 batches per
update](https://lightning.ai/docs/pytorch/stable/common/optimization.html#id3):

    yoyodyne-train --batch_size 1024 --accumulate_grad_batches 4 ...

The `--find_batch_size` flag enables [automatically computation of the batch
size](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html#batch-size-finder).
With `--find_batch_size max`, it simply uses the maximum batch size, ignoring
`--batch_size`. With `--find_batch_size opt`, it finds the maximum batch size,
and then interprets it as follows:

-   If the maximum batch size is greater than `--batch_size`, then
    `--batch_size` is used as the batch size.
-   However, if the maximum batch size is less than `--batch_size`, it solves
    for the optimal gradient accumulation trick and uses the largest batch size
    and the smallest number of gradient accumulation steps whose product is
    `--batch_size`.

If one wishes to solve for these quantities without actually training, pass
`--find_batch_size opt` and `--max_epochs 0`. This will halt after computing and
logging the solution.

### Hyperparameter tuning

**No neural model should be deployed without proper hyperparameter tuning.**
However, the default options give a reasonable initial settings for an attentive
biLSTM. For transformer-based architectures, experiment with multiple encoder
and decoder layers, much larger batches, and the warmup-plus-inverse square root
decay scheduler.

### Weights & Biases tuning

[`wandb_sweeps`](examples/wandb_sweeps) shows how to use [Weights &
Biases](https://wandb.ai/site) to run hyperparameter sweeps.

## Accelerators

[Hardware
accelerators](https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html)
can be used during training or prediction. In addition to CPU (the default) and
GPU (`--accelerator gpu`), [other
accelerators](https://pytorch-lightning.readthedocs.io/en/stable/extensions/accelerator.html)
may also be supported but not all have been tested yet.

## Precision

By default, training uses 32-bit precision. However, the `--precision` flag
allows the user to perform training with half precision (`16`) or with the
[`bfloat16` half precision
format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) if
supported by the accelerator. This may reduce the size of the model and batches
in memory, allowing one to use larger batches. Note that only default precision
is expected to work with CPU training.

## Examples

The [`examples`](examples) directory contains interesting examples, including:

-   [`wandb_sweeps`](examples/wandb_sweeps) shows how to use [Weights &
    Biases](https://wandb.ai/site) to run hyperparameter sweeps.

## For developers

*Developers, developers, developers!* - Steve Ballmer

This section contains instructions for the Yoyodyne maintainers.

### Design

Yoyodyne is beholden to the heavily object-oriented design of Lightning, and
wherever possible uses Torch to keep computations on the user-selected
accelerator. Furthermore, since it is developed at "low-intensity" by a
geographically-dispersed team, consistency is particularly important. Some
consistency decisions made thus far:

-   The abstract/concrete class distinction is enforced at runtime by the
    presence of methods raising `NotImplementedError`, though in the future we
    may consider [PEP 3119](https://peps.python.org/pep-3119/)-style
    "compile-time" enforcement.
-   [`numpy`](https://numpy.org/) is used for basic mathematical operations and
    constants even in places where the built-in
    [`math`](https://docs.python.org/3/library/math.html) would do.

#### Models and modules

A *model* in Yoyodyne is a sequence-to-sequence architecture and inherits from
`yoyodyne.models.BaseModel`. These models in turn consist of ("have-a") one or
two *encoders* responsible for building a numerical representation of the source
(and features, where appropriate) and a *decoder* responsible for predicting the
target sequence using the representation generated by the encoder. The encoders
and decoders themselves Torch modules inheriting from `torch.nn.Module`.

The model is responsible for constructing the encoders and decoders. The model
dictates the type of decoder; each model has a preferred encoder type as well,
though it may work with others. The model communicates with its modules by
calling them as functions (which invokes their `forward` methods); however, in
some cases it is also necessary for the model to call ancillary members or
methods of its modules. The `base.ModuleOutput` class is used to capture the
output of the various modules, and it is this which is essential to, e.g.,
abstracting between different kinds of encoders which may or may not have
hidden or cell state to return.

#### Decoding strategies

Each model supports greedy decoding implemented via a `greedy_decode` method;
some also support beam decoding via `beam_decode`. Some models (e.g., the hard
attention models) require teacher forcing, but most can be trained with either
student or teacher forcing.

### Releasing

1.  Create a new branch. E.g., if you want to call this branch "release":
    `git checkout -b release`
2.  Sync your fork's branch to the upstream master branch. E.g., if the upstream
    remote is called "upstream": `git pull upstream master`
3.  Increment the version field in [`pyproject.toml`](pyproject.toml).
4.  Stage your changes: `git add pyproject.toml`.
5.  Commit your changes: `git commit -m "your commit message here"`
6.  Push your changes. E.g., if your branch is called "release":
    `git push origin release`
7.  Submit a PR for your release and wait for it to be merged into `master`.
8.  Tag the `master` branch's last commit. The tag should begin with `v`; e.g.,
    if the new version is 3.1.4, the tag should be `v3.1.4`. This can be done:
    -   on GitHub itself: click the "Releases" or "Create a new release" link on
        the right-hand side of the Yoyodyne GitHub page) and follow the
        dialogues.
    -   from the command-line using `git tag`.
9.  Build the new release: `python -m build`
10. Upload the result to PyPI: `twine upload dist/*`

## References

Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., Grangier, D., and
Auli, M. 2019. [fairseq: a fast, extensible toolkit for sequence
modeling](https://aclanthology.org/N19-4009/). In *Proceedings of the 2019
Conference of the North American Chapter of the Association for Computational
Linguistics (Demonstrations)*, pages 48-53.

(See also [`yoyodyne.bib`](yoyodyne.bib) for more work used during the
development of this library.)
