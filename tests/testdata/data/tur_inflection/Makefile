SHELL := /bin/bash 

all: train.tsv \
	dev.tsv \
	test.tsv \
	context_hard_attention_lstm_separate_features_expected.test \
	context_hard_attention_lstm_separate_features_expected.txt \
	hard_attention_lstm_separate_features_expected.test \
	hard_attention_lstm_separate_features_expected.txt \
	pointer_generator_lstm_separate_features_expected.test \
	pointer_generator_lstm_separate_features_expected.txt \
	pointer_generator_transformer_linear_features_expected.test \
	pointer_generator_transformer_linear_features_expected.txt \
	soft_attention_lstm_gru_features_expected.test \
	soft_attention_lstm_gru_features_expected.txt \
	soft_attention_lstm_linear_features_expected.test \
	soft_attention_lstm_linear_features_expected.txt \
	soft_attention_lstm_separate_features_expected.test \
	soft_attention_lstm_separate_features_expected.txt \
	soft_attention_lstm_shared_features_expected.test \
	soft_attention_lstm_shared_features_expected.txt \
	soft_attention_lstm_separate_features_expected.test \
	soft_attention_lstm_separate_features_expected.txt \
	transformer_shared_features_expected.test \
	transformer_shared_features_expected.txt \
	transformer_invariant_features_expected.test \
	transformer_invariant_features_expected.txt

# Data from the CoNLL-SIGMORPHON 2017 shared task on morphological reinflection.

train.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/turkish-train-medium > "$@"

dev.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/turkish-dev > "$@"

test.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/answers/task1/turkish-uncovered-test > "$@"

%_expected.test %_expected.txt: train.tsv dev.tsv test.tsv ../../configs/%.yaml
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
        --trainer ../../configs/real_trainer.yaml \
        --trainer.default_root_dir "$(TMPDIR)"
	yoyodyne \
        test \
        --ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
        --data ../../configs/tur_inflection_data.yaml \
        --data.test test.tsv \
        --data.model_dir "$(TMPDIR)" \
        --model "../../configs/$*.yaml" \
        --trainer.default_root_dir "$(TMPDIR)" \
        --trainer.enable_progress_bar false \
        > "$*_expected.test"
	yoyodyne \
        predict \
        --ckpt_path "$(TMPDIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
        --data ../../configs/tur_inflection_data.yaml \
        --data.predict test.tsv \
        --data.model_dir "$(TMPDIR)" \
        --model "../../configs/$*.yaml" \
        --prediction.path "$*_expected.txt"
	rm -rf "$(TMPDIR)"
