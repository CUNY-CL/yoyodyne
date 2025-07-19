# Yoyodyne 🪀

[![PyPI
version](https://badge.fury.io/py/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![Supported Python
versions](https://img.shields.io/pypi/pyversions/yoyodyne.svg)](https://pypi.org/project/yoyodyne)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/CUNY-CL/yoyodyne/tree/master.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/CUNY-CL/yoyodyne/tree/master)

Yoyodyne provides small-vocabulary sequence-to-sequence generation with and
without feature conditioning.

These models are implemented using [PyTorch](https://pytorch.org/) and
[Lightning](https://www.pytorchlightning.ai/).

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
-   Models are specified using YAML configuration files.
-   Releases are made regularly and bugs addressed.
-   🚧 UNDER CONSTRUCTION 🚧: It has exhaustive test suites.
-   🚧 UNDER CONSTRUCTION 🚧: It has performance benchmarks.

## Authors

Yoyodyne was created by [Adam Wiemerslage](https://adamits.github.io/), [Kyle
Gorman](https://wellformedness.com/), Travis Bartley, and [other
contributors](https://github.com/CUNY-CL/yoyodyne/graphs/contributors) like
yourself.

## Installation

### Local installation

To install Yoyodyne and its dependencies, run the following command:

    pip install .

Then, optionally install additional dependencies for developers and testers:

    pip install -r requirements.txt

### Google Colab

Yoyodyne is also compatible with [Google
Colab](https://colab.research.google.com/) GPU runtimes.

1.  Click "Runtime" \> "Change Runtime Type".
2.  In the dialogue box, under the "Hardware accelerator" dropdown box, select
    "GPU", then click "Save".
3.  You may be prompted to delete the old runtime. Do so if you wish.
4.  Then install and run using the `!` as a prefix to shell commands.

## File formats

Other than YAML configuration files, Yoyodyne operates on basic tab-separated
values (TSV) data files. The user can specify source, features, and target
columns, and separators used to parse them.

### Data format

The default data format is a two-column TSV file in which the first column is
the source string and the second the target string.

    source   target

To enable the use of a features column, one specifies a (non-zero)
`data: features_col:` argument, and optionally also a `data: features_sep:`
argument (the default features separator is ";"). For instance, for the
[SIGMORPHON 2016 shared task](https://sigmorphon.github.io/sharedtasks/2016/)
data:

    source   feat1,feat2,...    target

the format is specified as:

    ...
    data:
        ...
        features_col: 2
        features_sep: ,
        target_col: 3
        ...

Alternatively, for the [CoNLL-SIGMORPHON 2017 shared
task](https://sigmorphon.github.io/sharedtasks/2017/), the first column is the
source (a lemma), the second is the target (the inflection), and the third
contains semi-colon delimited features strings:

    source   target    feat1;feat2;...

the format is specified as simply:

    ...
    data:
        ...
        features_col: 3

### Reserved symbols

Yoyodyne reserves symbols of the form `<...>` for internal use.
Feature-conditioned models also use `[...]` to avoid clashes between features
symbols and source and target symbols, and in some cases, `{...}` to avoid
clashes between source and target symbols. Therefore, users should not provide
any symbols of the form `<...>`, `[...]`, or `{...}`.

## Usage

The `yoyodyne` command-line tool uses a subcommand interface, with four
different modes. To see a full set of options available for each subcommand, use
the `--print_config` flag. For example:

    yoyodyne fit --print_config

will show all configuration options (and their default values) for the `fit`
subcommand.

For more detailed examples, see the [`configs`](configs) directory.

### Training (`fit`)

In `fit` mode, one trains a Yoyodyne model, either from scratch or, optionally,
resuming for a prexisting checkpoint. Naturally, most configuration options need
to be set at training time. E.g., it is not possible to switch modules after
training a model.

This mode is invoked using the `fit` subcommand, like so.

    yoyodyne fit --config path/to/config.yaml

Alternatively, one can resume training from a pre-existing checkpoint so long as
it matches the specification of the configuration file.

    yoyodyne fit --config path/to/config.yaml --ckpt path/to/checkpoint.ckpt

#### Seeding

Setting the `seed_everything:` argument to some fixed value ensures a
reproducible experiment (modulo hardware non-determism).

#### Model architecture

#### Optimization

Yoyodyne requires an optimizer and an learning rate scheduler. The default
optimizer is
[`torch.optim.Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html),
and the default scheduler is `yoyodyne.schedulers.Dummy`, which keeps learning
rate fixed at its initial value and takes no explicit configuration arguments.

The following YAML snippet shows the use of the Adam optimizer with a
non-default initial learning rate and the
`yoyodyne.schedulers.WarmupInverseSquareRoot` LR scheduler:

    ...
    model:
        ...
        optimizer:
          class_path: torch.optim.Adam
          init_args:
            lr: 1.0e-5 
        scheduler:
          class_path: yoyodyne.schedulers.WarmupInverseSquareRoot
          init_args:
            warmup_epochs: 10   
        ...

#### Checkpointing

The
[`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
is used to control the generation of checkpoint files. A sample YAML snippet is
given below.

    ...
    checkpoint:
      filename: "model-{epoch:03d}-{val_accuracy:.4f}"
      mode: max
      monitor: val_accuracy
      verbose: true
      ...

Alternatively, one can specify a checkpointing that minimizes validation loss,
as follows.

    ...
    checkpoint:
      filename: "model-{epoch:03d}-{val_loss:.4f}"
      mode: min
      monitor: val_loss
      verbose: true
      ...

A checkpoint config must be specified or Yoyodyne will not generate any
checkpoints.

#### Callbacks

The user will likely want to configure additional callbacks. Some useful
examples are given below.

The
[`LearningRateMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html)
callback records learning rates:

    ...
    trainer:
      callbacks:
      - class_path: lightning.pytorch.callbacks.LearningRateMonitor
        init_args:
          logging_interval: epoch
      ...

The
[`EarlyStopping`](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)
callback enables early stopping based on a monitored quantity and a fixed
`patience`:

    ...
    trainer:
      callbacks:
      - class_path: lightning.pytorch.callbacks.EarlyStopping
        init_args:
          monitor: val_loss
          patience: 10
          verbose: true
      ...

#### Logging

By default, Yoyodyne performs some minimal logging to standard error and uses
progress bars to keep track of progress during each epoch. However, one can
enable additional logging faculties during training, using a similar syntax to
the one we saw above for callbacks.

The
[`CSVLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html)
logs all monitored quantities to a CSV file:

    ...
    trainer:
      logger:
        - class_path: lightning.pytorch.loggers.CSVLogger
          init_args:
            save_dir: /Users/Shinji/models
      ...
       

The
[`WandbLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html)
works similarly to the `CSVLogger`, but sends the data to the third-party
website [Weights & Biases](https://wandb.ai/site), where it can be used to
generate charts or share artifacts:

    ...
    trainer:
      logger:
      - class_path: lightning.pytorch.loggers.WandbLogger
        init_args:
          project: unit1
          save_dir: /Users/Shinji/models
      ...

Note that this functionality requires a working account with Weights & Biases.

#### Other options

Dropout probability and/or label smoothing are specified as arguments to the
`model` and its encoders:

    ...
    model:
      source_encoder:
        class_path: ...
        init_args: ...
          dropout: 0.5
      decoder_dropout: 0.5
      label_smoothing: 0.1
      ...

Batch size is specified using `data: batch_size: ...` and defaults to 32.

By default, the source and target vocabularies share embeddings so identical
source and target symbols will have the same embedding. This can be disabled
with `data: tie_embeddings: false`.

By default, training uses 32-bit precision. However, the `trainer.: precision:`
flag allows the user to perform training with half precision (`16`), or with
mixed-precision formats like `bf16-mixed` if supported by the accelerator. This
might reduce the size of the model and batches in memory, allowing one to use
larger batches, or it may simply provide small speed-ups.

There are a number of ways to specify how long a model should train for. For
example, the following YAML snippet specifies that training should run for 100
epochs or 6 wall-clock hours, whichever comes first:

    ...
    trainer:
      max_epochs: 100
      max_time: 00:06:00:00
      ...

### Validation (`validate`)

In `validation` mode, one runs the validation step over labeled validation data
(specified as `data: val: path/to/validation.tsv`) using a previously trained
checkpoint (`--ckpt_path path/to/checkpoint.ckpt` from the command line),
recording loss and other statistics for the validation set. In practice this is
mostly useful for debugging.

This mode is invoked using the `validate` subcommand, like so:

    yoyodyne validate --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

### Evaluation (`test`)

In `test` mode, one computes accuracy over held-out test data (specified as
`data: test: path/to/test.tsv`) using a previously trained checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line); it differs from
validation mode in that it uses the `test` file rather than the `val` file.

This mode is invoked using the `test` subcommand, like so:

    yoyodyne test --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

#### Inference (`predict`)

In `predict` mode, a previously trained model checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line) is used to label
an input file. One must also specify the path where the predictions will be
written:

    ...
    predict:
      path: /Users/Shinji/predictions.txt
    ...

This mode is invoked using the `predict` subcommand, like so:

    yoyodyne predict --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

Vanilla RNN models (like `yoyodyne.models.AttentiveGRUModel` or
`yoyodyne.models.AttentiveLSTMModel`) and pointer-generator RNN models (like
`yoyodyne.models.PointerGeneratorGRUModel` or
`yoyodyne.models.PointerGeneratorLSTMModel`) support beam search during
prediction. This is enabled by setting a `beam_width` \> 1, but also requires a
`batch_size` of 1:

    data:
      ...
      batch_size: 1
      ...
    model:
      class_path: yoyodyne.models.AttentiveLSTMModel
      init_args:
        ...
        beam_width: 5
        ...
    predict:
      path: /Users/Shinji/predictions.tsv
      ...

The resulting prediction files will be a 10-column TSV file consisting of the
top 5 target hypotheses and their log-likelihoods (collated together), rather
than single-file text files just containing the top hypothesis.

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
    instead uses an LSTM decoder and encoder (by default). `--attention_context`
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
    a neural transducer mechanism. One must pass an edit distance transducer
    learned using expectation using the
    [`maxwell`](https://github.com/CUNY-CL/maxwell) library; see [the example
    here](examples/maxwell) for more information. Imitation learning is used to
    train the model to to implement the oracle policy, with roll-in controlled
    by the `--oracle_factor` flag (default: `1`). Since this model assumes
    monotonic alignment, it may be superior to attentive models when the
    alignment between input and output is roughly monotonic and when input and
    output vocabularies overlap significantly.
-   `transducer_lstm`: This is similar to the `transducer_gru` but instead uses
    an LSTM decoder and encoder (by default).
-   `transformer`: This is a transformer decoder with transformer encoders (by
    default). Sinusodial positional encodings and layer normalization are used.
    The user may wish to specify the number of attention heads (with
    `--attention_heads`; default: `4`).

The `--arch` flag specifies the decoder type; the user can override default
encoder types using the `--source_encoder_arch` flag and, when features are
present, the `--features_encoder_arch` flag. Valid values are:

-   `feature_invariant_transformer` (usually used with
    `--features_encoder_arch`): a variant of the transformer encoder used with
    features; it concatenates source and features and uses a learned embedding
    to distinguish between source and features symbols.
-   `linear` (usually used with `--features_encoder_arch`): a non-contextual
    encoder with a affine transformation applied to embeddings
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

## Examples

The [`examples`](examples) directory contains interesting examples, including:

-   [`concatenate`](examples/concatenate) provides sample code for concatenating
    source and features symbols à la Kann & Schütze (2016).
-   [`wandb_sweeps`](examples/wandb_sweeps) shows how to use [Weights &
    Biases](https://wandb.ai/site) to run hyperparameter sweeps.

## Related projects

-   [Maxwell](https://github.com/CUNY-CL/maxwell) is used to learn a stochastic
    edit distance model for the neural transducer.
-   [Yoyodyne Pretrained](https://github.com/CUNY-CL/yoyodyne-pretrained)
    provides a similar interface but uses large pre-trained models to initialize
    the encoder and decoder modules.

## Testing

🚧 UNDER CONSTRUCTION 🚧

### License

Yoyodyne is distributed under an [Apache 2.0 license](LICENSE.txt).

## For developers

We welcome contributions using the fork-and-pull model.

### Design

Yoyodyne is beholden to the heavily object-oriented design of Lightning, and
wherever possible uses Torch to keep computations on the user-selected
accelerator. Furthermore, since it is developed at "low-intensity" by a
geographically-dispersed team, consistency is particularly important. Some
consistency decisions made thus far:

-   Abstract classes overrides are enforced using [PEP
    3119](https://peps.python.org/pep-3119/).
-   [`numpy`](https://numpy.org/) is used for basic mathematical operations and
    constants even in places where the built-in
    [`math`](https://docs.python.org/3/library/math.html) would do.

#### Models and modules

A *model* in Yoyodyne is a sequence-to-sequence architecture and inherits from
`yoyodyne.models.BaseModel`. These models in turn consist of ("have-a") one or
more *encoders* responsible for encoding the source (and features, where
appropriate), and a *decoder* responsible for predicting the target sequence
using the representation generated by the encoders. The encoders and decoder are
themselves Torch modules.

The model is responsible for constructing the encoders and decoders. The model
dictates the type of decoder. The model communicates with its modules by calling
them as functions (which invokes their `forward` methods); however, in some
cases it is also necessary for the model to call ancillary members or methods of
its modules.

When features are present, models are responsible for fusing source and features
encodings, and do so in a model-specific fashion. For example, ordinary RNNs and
transformers concatenate source and features encodings on the length dimension
(and thus require that the encodings be the same size), whereas hard attention
and transducer models average across the features encoding across the length
dimension and the concatenate the resulting tensor with the source encoding on
the encoding dimension; by doing so they preserve the source length and make it
impossible to attend directly to features symbols.

#### Decoding strategies

Each model supports greedy decoding implemented via a `greedy_decode` method;
some also support beam decoding via `beam_decode`
(cf. [#17](https://github.com/CUNY-CL/yoyodyne/issues/17)).

Some models (e.g., the hard attention models) require teacher forcing, but
others can be trained with either student or teacher forcing
(cf. [#77](https://github.com/CUNY-CL/yoyodyne/issues/77)).

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

Kann, K. and Schütze, H. 2016. [Single-model encoder-decoder with explicit
morphological representation for
reinflection](https://aclanthology.org/P16-2090/). In *Proceedings of the 54th
Annual Meeting of the Association for Computational Linguistics (Volume 2: Short
Papers)*, pages 555-560.

Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., Grangier, D., and
Auli, M. 2019. [fairseq: a fast, extensible toolkit for sequence
modeling](https://aclanthology.org/N19-4009/). In *Proceedings of the 2019
Conference of the North American Chapter of the Association for Computational
Linguistics (Demonstrations)*, pages 48-53.

(See also [`yoyodyne.bib`](yoyodyne.bib) for more work used during the
development of this library.)
