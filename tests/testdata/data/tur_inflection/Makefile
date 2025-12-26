SHELL := /bin/bash

all: train.tsv \
	dev.tsv \
	test.tsv \
	attentive_gru_expected.test \
	attentive_lstm_expected.test \
    attentive_lstm_gru_features_expected.test \
    attentive_lstm_linear_features_expected.test \
    attentive_lstm_separate_features_expected.test \
    attentive_lstm_shared_features_expected.test \
	attentive_lstm_gru_source_expected.test \
	attentive_lstm_transformer_source_expected.test \
	context_hard_attention_gru_expected.test \
	context_hard_attention_lstm_expected.test \
	gru_expected.test \
	hard_attention_gru_expected.test \
	hard_attention_lstm_expected.test \
	lstm_expected.test \
	pointer_generator_gru_expected.test \
	pointer_generator_lstm_expected.test \
	pointer_generator_transformer_expected.test \
	transformer_expected.test

# Data from the CoNLL-SIGMORPHON 2017 shared task on morphological reinflection.

train.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/turkish-train-medium > "$@"

dev.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/turkish-dev > "$@"

test.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/answers/task1/turkish-uncovered-test > "$@"

%_expected.test: train.tsv dev.tsv test.tsv ../../configs/%.yaml
	$(eval TMPDIR := $(shell mktemp -d))
	yoyodyne \
		fit \
		--checkpoint ../../configs/checkpoint.yaml \
		--data ../../configs/tur_inflection_data.yaml \
		--data.train train.tsv \
		--data.val dev.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../../configs/$*.yaml" \
		--seed_everything 49 \
		--trainer ../../configs/real_trainer.yaml 
	yoyodyne \
		test \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../../configs/tur_inflection_data.yaml \
		--data.test test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../../configs/$*.yaml" \
		--trainer.enable_progress_bar false \
		> "$@"
	rm -rf "$(TMPDIR)"
