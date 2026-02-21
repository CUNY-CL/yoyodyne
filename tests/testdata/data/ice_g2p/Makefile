SHELL := /bin/bash

all: train.tsv \
	dev.tsv \
	test.tsv \
	context_hard_attention_gru_expected.test \
	context_hard_attention_gru_expected.txt \
	context_hard_attention_lstm_expected.test \
	context_hard_attention_lstm_expected.txt \
	gru_expected.test \
	gru_expected.txt \
	hard_attention_gru_expected.test \
	hard_attention_gru_expected.txt \
	hard_attention_lstm_expected.test \
	hard_attention_lstm_expected.txt \
	lstm_expected.test \
	lstm_expected.txt \
	pointer_generator_gru_expected.test \
	pointer_generator_gru_expected.txt \
	pointer_generator_lstm_expected.test \
	pointer_generator_lstm_expected.txt \
	pointer_generator_lstm_student_forcing_expected.test \
	pointer_generator_lstm_student_forcing_expected.txt \
	soft_attention_gru_expected.test \
	soft_attention_gru_expected.txt \
	soft_attention_lstm_expected.test \
	soft_attention_lstm_expected.txt \
	soft_attention_lstm_student_forcing_expected.test \
	soft_attention_lstm_student_forcing_expected.txt \
	soft_attention_lstm_gru_source_expected.test \
	soft_attention_lstm_gru_source_expected.txt \
	soft_attention_lstm_transformer_source_expected.test \
	soft_attention_lstm_transformer_source_expected.txt \
	transformer_expected.test \
	transformer_expected.txt

# Data from the SIGMORPHON 2021 shared task on grapheme-to-phoneme conversion.

train.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/2021-task1/refs/heads/main/data/low/ice_train.tsv > "$@"

dev.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/2021-task1/refs/heads/main/data/low/ice_dev.tsv > "$@"

test.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/2021-task1/refs/heads/main/data/low/ice_test.tsv > "$@"

%_expected.test %_expected.txt: train.tsv dev.tsv test.tsv ../../configs/%.yaml
	$(eval TMPDIR := $(shell mktemp -d))
	yoyodyne \
		fit \
		--checkpoint ../../configs/checkpoint.yaml \
		--data ../../configs/ice_g2p_data.yaml \
		--data.train train.tsv \
		--data.val dev.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../../configs/$*.yaml" \
		--seed_everything 49 \
		--trainer ../../configs/real_trainer.yaml \
		--trainer.default_root_dir "$(TMPDIR)"
	yoyodyne \
		test \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../../configs/ice_g2p_data.yaml \
		--data.test test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../../configs/$*.yaml" \
		--trainer.default_root_dir "$(TMPDIR)" \
		--trainer.enable_progress_bar false \
		> "$*_expected.test"
	yoyodyne \
		predict \
		--ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../../configs/ice_g2p_data.yaml \
		--data.predict test.tsv \
		--data.model_dir "$(TMPDIR)" \
		--model "../../configs/$*.yaml" \
		--prediction.path "$*_expected.txt"
	rm -rf "$(TMPDIR)"
