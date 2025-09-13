SHELL := /bin/bash

all: train.tsv \
	dev.tsv \
	test.tsv \
	attentive_gru_expected.test \
	attentive_lstm_expected.test \
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

%.tsv:
	$(eval TMPDIR := $(shell mktemp -d))
	.././randgen.py --vocabulary 26 --length 5 --seed 1951 --size 100 > "$(TMPDIR)/copy_$@"
	paste "$(TMPDIR)/copy_$@" <(paste -d'\0' "$(TMPDIR)/copy_$@" "$(TMPDIR)/copy_$@") > "$@"
	rm -rf "$(TMPDIR)"

%_expected.test: train.tsv dev.tsv test.tsv
	$(eval TMPDIR := $(shell mktemp -d))
	yoyodyne \
		fit \
		--checkpoint ../../configs/checkpoint.yaml \
		--data ../../configs/toy_data.yaml \
		--data.train train.tsv \
		--data.val dev.tsv \
		--data.model_dir $(TMPDIR) \
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
