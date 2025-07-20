# Sample configuration files

The files in this directory are example YAML configuration files showing
Yoyodyne's use with various architectures.

`LightningCLI` can:

-   combine options from multiple top-level configuration files:

        yoyodyne fit --config /path/to/config1.yaml --config /path/to/config2.yaml ...

-   combine options from multiple lower-level configuration files:

        yoyodyne fit --trainer /path/to/trainer_config.yaml --model /path/to/model_config.yaml ...

-   override options specified in configuration files using command-line flags:

        yoyodyne fit --config /path/to/config.yaml --data.batch_size 1024 ...

-   create a single combined configuration file (NB: won't actually train):

        yoyodyne fit --trainer /path/to/trainer_config.yaml --model /path/to/model_config.yaml --print_config=skip_default,skip_null

The `LightningCLI` configuration system is extremely powerful and advanced users
may want to read the docs
[here](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html)
before beginning.

## Trainer configurations

-   [`trainer.yaml`](trainer.yaml) specifies some commonly used trainer options,
    including:
    -   gradient clipping
    -   the maximum number of epochs
    -   the maximum "wall clock" time for training
    -   callbacks for monitoring learning rate and triggering early stopping
    -   local CSV logging
    -   remote [Weights & Biases](https://wandb.ai/site) logging

To use, run:

    yoyodyne fit --trainer /path/to/trainer_config.yaml ...

## Checkpointing configurations

-   [`accuracy_checkpoint.yaml`](accuracy_checkpoint.yaml) saves the single
    checkpoint with the highest validation accuracy.
-   [`loss_checkpoint.yaml`](loss_checkpoint.yaml) saves the single checkpoint
    with the lowest validation loss.

To use, run:

    yoyodyne fit --checkpoint /path/to/checkpoint_config.yaml ...

## Data configurations

-   [`sigmorphon2016.yaml`](sigmorphon2016.yaml) specifies the data format for
    the [SIGMORPHON 2016 shared
    task](https://sigmorphon.github.io/sharedtasks/2016/) data:

        source feat1,feat2,... target

To use, run:

    yoyodyne fit --data /path/to/data_config.yaml ...

## Model configurations

To use any of the configurations below, run:

    yoyodyne fit --model /path/to/model_config.yaml ...

### Feature-free configurations

These models all involve a simple sequence-to-sequence transduction without any
feature conditioning.

-   [`attentive_gru.yaml`](attentive_gru.yaml) and
    [`attentive_lstm.yaml`](attentive_lstm.yaml) are attentive RNN models with
    hyperparameters similar to those of MED (Kann & Schütze 2016).
-   [`attentive_unilstm.yaml`](attentive_unilstm.yaml) is a variant of
    [`attentive_lstm.yaml`](attentive_lstm.yaml) with a uni- (rather than bi-)
    directional encoder. Performance is often quite a bit worse with this
    design.
-   [`attentive_gru2lstm.yaml`](attentive_gru2lstm.yaml) is a MED-like variant
    where the encoder is an GRU but the decoder is a LSTM.
-   [`attentive_deep_lstm.yaml`](attentive_deep_lstm.yaml) is a MED-like variant
    where the encoder has 2 LSTM layers and the decoder has 2 LSTM layers.
    Performance is often quite a bit worse with this design.
-   [`frankenformer.yaml`](frankenformer.yaml) is a hybrid system with a
    transformer encoder and an attentive LSTM decoder. This model is in need of
    hyperparameter tuning before it is ready for deployment.
-   [`hard_attention_lstm.yaml`](hard_attention_lstm.yaml) and
    [`context_hard_attention_lstm.yaml`](context_hard_attention_lstm.yaml) are
    variants of hard attention LSTMs with hyperparameters similar to those of Wu
    & Cotterell (2019).
-   [`pointer_generator_lstm.yaml`](pointer_generator_lstm.yaml) shows a
    pointer-generator LSTM with hyperparameters similar to those of Sharma et
    al. (2018, their "high setting").
-   [`pointer_generator_transformer.yaml`](pointer_generator_transformer.yaml)
    shows a pointer-generator transformer with hyperparameters similar to those
    of Singer & Kann (2020).
-   [`transducer_lstm.yaml`](transducer_lstm.yaml) shows a transducer LSTM with
    hyperparameters similar to those used by Makarov & Clematide (2021).
-   [`transformer.yaml`](transformer.yaml) is a transformer with hyperparameters
    similar to those of Wu et al. (2021, "A smaller Transformer").

## Feature-conditioned configurations

Most of the patterns from above generalize to feature-conditioned transduction,
but we illustrate different ways one can enable feature conditioning.

-   [`attentive_lstm_shared.yaml`](attentive_lstm_shared.yaml) is a MED-like
    variant with a shared source and features encoder.
-   [`attentive_lstm_separate.yaml`](attentive_lstm_separate.yaml) is a MED-like
    variant with a separate LSTM features encoder.
-   [`attentive_lstm_linear.yaml`](attentive_lstm_linear.yaml) is a MED-like
    variant with a separate linear features encoder.
-   [`transformer_shared.yaml`](transformer_shared.yaml) is a transformer with a
    shared source and features encoder.

## Putting it all together

[`algother.yaml`](altogether.yaml) is a single combined configuration file with
trainer, checkpoint, data, model, and prediction configuration information.

To use, run:

    yoyodyne fit --config /path/to/config.yaml
