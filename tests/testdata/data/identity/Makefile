SHELL := /bin/bash

all: train.tsv \
	dev.tsv \
	test.tsv \
	soft_attention_gru_expected.test \
	soft_attention_lstm_expected.test \
	soft_attention_lstm_gru_source_expected.test \
	soft_attention_lstm_transformer_source_expected.test \
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

%.tsv:
	$(eval TMPDIR := $(shell mktemp -d))
	.././randgen.py --vocabulary 26 --length 5 --seed 1954 --size 100 > "$(TMPDIR)/identity_$@"
	paste "$(TMPDIR)/identity_$@" "$(TMPDIR)/identity_$@" > "$@"
	rm -rf "$(TMPDIR)"

%_expected.test: train.tsv dev.tsv test.tsv ../../configs/%.yaml
	$(eval TMPDIR := $(shell mktemp -d))
	yoyodyne \
		fit \
		--checkpoint ../../configs/checkpoint.yaml \
		--data ../../configs/toy_data.yaml \
		--data.train train.tsv \
		--data.val dev.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../../configs/$*.yaml" \
		--seed_everything 49 \
		--trainer ../../configs/toy_trainer.yaml 
	yoyodyne \
		test \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../../configs/toy_data.yaml \
		--data.test test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../../configs/$*.yaml" \
		--trainer.enable_progress_bar false \
		> "$@"
	rm -rf "$(TMPDIR)"
