# Yoyodyne 🪀

[![PyPI
version](https://badge.fury.io/py/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![CircleCI](https://circleci.com/gh/CUNY-CL/yoyodyne/tree/master.svg?style=svg&circle-token=37883deeb03d32c8a7b2aa7c34e5143bf514acdd?)](https://circleci.com/gh/CUNY-CL/yoyodyne/tree/master)

Yoyodyne provides neural models for small-vocabulary sequence-to-sequence
generation with and without feature conditioning.

These models are implemented using [PyTorch](https://pytorch.org/) and
[Lightning](https://www.pytorchlightning.ai/).

While we provide classic LSTM and transformer models, some of the provided
models are particularly well-suited for problems where the source-target
alignments are roughly monotonic (e.g., `transducer`) and/or where source and
target vocabularies have substantial overlap (e.g., `pointer_generator_lstm`).

## Philosophy

Yoyodyne is inspired by [FairSeq](https://github.com/facebookresearch/fairseq)
(Ott et al. 2019) but differs on several key points of design:

-   It is for small-vocabulary sequence-to-sequence generation, and therefore
    includes no affordances for machine translation or language modeling.
    Because of this:
    -   It has no plugin interface and the architectures provided are intended
        to be reasonably exhaustive.
    -   There is little need for data preprocessing; it works with TSV files.
-   It has support for using features to condition decoding, with
    architecture-specific code to handle feature information.
-   It uses validation accuracy (not loss) for model selection and early
    stopping.
-   Releases are made regularly.
-   🚧 UNDER CONSTRUCTION 🚧: It has exhaustive test suites.
-   🚧 UNDER CONSTRUCTION 🚧: It has performance benchmarks.

## Installation

### Local installation

Yoyodyne currently supports Python 3.9 and 3.10.
[#60](https://github.com/CUNY-CL/yoyodyne/issues/60) is a known blocker to
Python \> 3.10 support.

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
-   `--experiment`: name of experiment (pick something unique)
-   `--train`: path to TSV file containing training data
-   `--val`: path to TSV file containing validation data

The user can also specify various optional training and architectural arguments.
See below or run [`yoyodyne-train --help`](yoyodyne/train.py) for more
information.

### Prediction

Prediction is performed by the [`yoyodyne-predict`](yoyodyne/predict.py) script.
One must specify the following required arguments:

-   `--model_dir`: path for model metadata
-   `--experiment`: name of experiment
-   `--checkpoint`: path to checkpoint
-   `--predict`: path to file containing data to be predicted
-   `--output`: path for predictions

The `--predict` file can either be a TSV file or an ordinary TXT file with one
source string per line; in the latter case, specify `--target_col 0`. Run
[`yoyodyne-predict --help`](yoyodyne/predict.py) for more information.

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
symbols and source and target symbols. Therefore, users should not provide any
symbols of the form `<...>` or `[...]`.

## Model checkpointing

Checkpointing is handled by
[Lightning](https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html).
The path for model information, including checkpoints, is specified by a
combination of `--model_dir` and `--experiment`, such that we build the path
`model_dir/experiment/version_n`, where each run of an experiment with the same
`model_dir` and `experiment` is namespaced with a new version number. A version
stores everything needed to reload the model, including the hyperparameters
(`model_dir/experiment_name/version_n/hparams.yaml`) and the checkpoints
directory (`model_dir/experiment_name/version_n/checkpoints`).

By default, each run initializes a new model from scratch, unless the
`--train_from` argument is specified. To continue training from a specific
checkpoint, the **full path to the checkpoint** should be specified with for the
`--train_from` argument. This creates a new version, but starts training from
the provided model checkpoint.

During training, we save the best `--save_top_k` checkpoints (by default, 1)
ranked according to accuracy on the `--val` set. For example, `--save_top_k 5`
will save the top 5 most accurate models.

## Models

The user specifies the overall architecture for the model using the `--arch`
flag. The value of this flag specifies the decoder's architecture and whether or
not an attention mechanism is present. This flag also specifies a default
architecture for the encoder(s), but it is possible to override this with
additional flags. Supported values for `--arch` are:

-   `attentive_lstm`: This is an LSTM decoder with LSTM encoders (by default)
    and an attention mechanism. The initial hidden state is treated as a learned
    parameter.
-   `lstm`: This is an LSTM decoder with LSTM encoders (by default); in lieu of
    an attention mechanism, the last non-padding hidden state of the encoder is
    concatenated with the decoder hidden state.
-   `pointer_generator_lstm`: This is an LSTM decoder with LSTM encoders (by
    default) and a pointer-generator mechanism. Since this model contains a copy
    mechanism, it may be superior to an ordinary attentive LSTM when the source
    and target vocabularies overlap significantly. Note that this model requires
    that the number of `--encoder_layers` and `--decoder_layers` match.
-   `pointer_generator_transformer`: This is a transformer decoder with
    transformer encoders (by default) and a pointer-generator mechanism. Like
    `pointer_generator_lstm`, it may be superior to an ordinary transformer when
    the source and target vocabularies overlap significantly. When using
    features, the user may wish to specify the n umber of features attention
    heads (with `--features_attention_heads`; default: `1`).
-   `transducer`: This is an LSTM decoder with LSTM encoders (by default) and a
    neural transducer mechanism. On model creation, expectation maximization is
    used to learn a sequence of edit operations, and imitation learning is used
    to train the model to implement the oracle policy, with roll-in controlled
    by the `--oracle_factor` flag (default: `1`). Since this model assumes
    monotonic alignment, it may be superior to attentive models when the
    alignment between input and output is roughly monotonic and when input and
    output vocabularies overlap significantly.
-   `transformer`: This is a transformer decoder with transformer encoders (by
    default). Sinusodial positional encodings and layer normalization are used.
    The user may wish to specify the number of attention heads (with
    `--source_attention_heads`; default: `4`).

The user can override the default encoder architectures. One can override the
source encoder using the `--source_encoder` flag:

-   `feature_invariant_transformer`: This is a variant of the transformer
    encoder used with features; it concatenates source and features and uses a
    learned embedding to distinguish between source and features symbols.
-   `linear`: This is a linear encoder.
-   `lstm`: This is a LSTM encoder.
-   `transformer`: This is a transformer encoder.

When using features, the user can also specify a non-default features encoder
using the `--features_encoder` flag (`linear`, `lstm`, `transformer`).

For all models, the user may also wish to specify:

-   `--decoder_layers` (default: `1`): number of decoder layers
-   `--embedding` (default: `128`): embedding size
-   `--encoder_layers` (default: `1`): number of encoder layers
-   `--hidden_size` (default: `512`): hidden layer size

By default, LSTM encoders are bidirectional. One can disable this with the
`--no_bidirectional` flag.

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
    -   `--patience`
-   Seeding:
    -   `--seed`
-   [Weights & Biases](https://wandb.ai/site):
    -   `--log_wandb` (default: `False`): enables Weights & Biases tracking

### Schedulers

By default, Yoyodyne uses a constant learning rate during training, but best
practice is to gradually decreasing learning rate as the model approaches
convergence using a [scheduler](yoyodyne/schedulers.py). Three (non-null)
schedulers are supported and are selected with `--scheduler`:

-   `lineardecay`: linearly decreases the learning rate (multiplying it by
    `--start_factor`) for `--total_decay_steps` steps, then decreases the
    learning rate by `--end_factor`.
-   `reduceonplateau`: reduces the learning rate (multiplying it by
    `--reduceonplateau_factor`) after `--reduceonplateau_patience` epochs with
    no improvement (when the loss stops decreasing if `--reduceonplateau loss`,
    or when the validation accuracy stops increasing if
    `--reduceonplateaumode accuracy`) until the learning rate is less than or
    equal to `--min_learning_rate`.
-   `warmupinvsqrt`: linearly increases the learning rate from 0 to
    `--learning_rate` for `--warmup_steps` steps, then decreases learning rate
    according to an inverse root square schedule.

### Simulating large batches

At times one may wish to train with a larger batch size than will fit in "in
core". For example, suppose one wishes to fit with a batch size of 4,096, but
this gives an out of memory exception. Then, with minimal overhead, one could
simulate an effective batch size of 4,096 by using batches (`--batch_size`) of
1,024, accumulating gradients from 4 batches
([`--accumulate_grad_batches`](https://lightning.ai/docs/pytorch/stable/common/optimization.html#id3))
per update:

    yoyodyne-train --batch_size 1024 --accumulate_grad_batches 4 ...

### Automatic tuning

`yododyne-train --auto_lr_find` uses a heuristic ([Smith
2017](https://ieeexplore.ieee.org/abstract/document/7926641)) to propose an
initial learning rate. Batch auto-scaling is not supported.

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
in memory, allowing one to use larger batches.

## Examples

The [`examples`](examples) directory contains interesting examples, including:

-   [`wandb_sweeps`](examples/wandb_sweeps) shows how to use [Weights &
    Biases](https://wandb.ai/site) to run hyperparameter sweeps.

## For developers

*Developers, developers, developers!* - Steve Ballmer

This section contains instructions for the Yoyodyne maintainers.

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

Smith, L. N. 2017. [Cyclical learning rates for training neural
networks](https://ieeexplore.ieee.org/abstract/document/7926641). In *2017 IEEE
Winter Conference on Applications of Computer Vision*, pages 464-472.
