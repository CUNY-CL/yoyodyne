SHELL := /bin/bash

all: train.tsv \
	dev.tsv \
	test.tsv \
	train.sed \
	causal_transformer_expected.test \
	causal_transformer_expected.txt \
	causal_transformer_beam_expected.txt \
	causal_transformer_student_forcing_expected.test \
	causal_transformer_student_forcing_expected.txt \
	context_hard_attention_gru_expected.test \
	context_hard_attention_gru_expected.txt \
	context_hard_attention_lstm_expected.test \
	context_hard_attention_lstm_expected.txt \
	gru_expected.test \
	gru_expected.txt \
	gru_beam_expected.txt \
	hard_attention_gru_expected.test \
	hard_attention_gru_expected.txt \
	hard_attention_lstm_expected.test \
	hard_attention_lstm_expected.txt \
	lstm_expected.test \
	lstm_expected.txt \
	lstm_beam_expected.txt \
	pointer_generator_gru_expected.test \
	pointer_generator_gru_expected.txt \
	pointer_generator_gru_beam_expected.txt \
	pointer_generator_lstm_expected.test \
	pointer_generator_lstm_expected.txt \
	pointer_generator_lstm_beam_expected.txt \
	pointer_generator_lstm_student_forcing_expected.test \
	pointer_generator_lstm_student_forcing_expected.txt \
	pointer_generator_transformer_expected.test \
	pointer_generator_transformer_expected.txt \
	pointer_generator_transformer_beam_expected.txt \
	pointer_generator_transformer_student_forcing_expected.test \
	pointer_generator_transformer_student_forcing_expected.txt \
	rotary_causal_transformer_expected.test \
	rotary_causal_transformer_expected.txt \
	rotary_pointer_generator_transformer_expected.test \
	rotary_pointer_generator_transformer_expected.txt \
	rotary_transformer_expected.test \
	rotary_transformer_expected.txt \
	soft_attention_gru_expected.test \
	soft_attention_gru_expected.txt \
	soft_attention_gru_beam_expected.txt \
	soft_attention_lstm_expected.test \
	soft_attention_lstm_expected.txt \
	soft_attention_lstm_beam_expected.txt \
	soft_attention_lstm_student_forcing_expected.test \
	soft_attention_lstm_student_forcing_expected.txt \
	soft_attention_lstm_gru_source_expected.test \
	soft_attention_lstm_gru_source_expected.txt \
	soft_attention_lstm_transformer_source_expected.test \
	soft_attention_lstm_transformer_source_expected.txt \
	transformer_expected.test \
	transformer_expected.txt \
	transformer_beam_expected.txt \
	transformer_absolute_positional_expected.test \
	transformer_absolute_positional_expected.txt \
	transformer_student_forcing_expected.test \
	transducer_gru_expected.test \
	transducer_gru_expected.txt \
	transducer_lstm_expected.test \
	transducer_lstm_expected.txt \
	transducer_lstm_student_forcing_expected.test \
	transducer_lstm_student_forcing_expected.txt

%.tsv:
	$(eval TMPDIR := $(shell mktemp -d))
	.././randgen.py --vocabulary 26 --length 5 --seed 1951 --size 100 > "$(TMPDIR)/copy_$@"
	paste "$(TMPDIR)/copy_$@" <(paste -d'\0' "$(TMPDIR)/copy_$@" "$(TMPDIR)/copy_$@") > "$@"
	rm -rf "$(TMPDIR)"

train.sed: train.tsv
	maxwell-train --train train.tsv --output "$@"

%_expected.test %_expected.txt: train.tsv dev.tsv test.tsv train.sed ../../configs/%.yaml
	$(eval MODEL_DIR := $(CURDIR)/models/$*)
	$(eval SED_ARG := $(if $(filter transducer%,$*),--model.sed_path train.sed,))
	mkdir -p "$(MODEL_DIR)"
	yoyodyne \
		fit \
		--checkpoint ../../configs/checkpoint.yaml \
		--data ../../configs/toy_data.yaml \
		--data.model_dir "$(MODEL_DIR)" \
		--data.train train.tsv \
		--data.val dev.tsv \
		--model "../../configs/$*.yaml" \
		$(SED_ARG) \
		--seed_everything 49 \
		--trainer ../../configs/toy_trainer.yaml \
		--trainer.default_root_dir "$(MODEL_DIR)"
	yoyodyne \
		test \
		--ckpt_path "$(MODEL_DIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../../configs/toy_data.yaml \
		--data.model_dir "$(MODEL_DIR)" \
		--data.test test.tsv \
		--model "../../configs/$*.yaml" \
		$(SED_ARG) \
		--trainer.default_root_dir "$(MODEL_DIR)" \
		--trainer.enable_progress_bar false \
		> "$*_expected.test"
	yoyodyne \
		predict \
		--ckpt_path "$(MODEL_DIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../../configs/toy_data.yaml \
		--data.model_dir "$(MODEL_DIR)" \
		--data.predict test.tsv \
		--model "../../configs/$*.yaml" \
		$(SED_ARG) \
		--prediction.path "$*_expected.txt"

%_beam_expected.txt: %_expected.txt
	$(eval BASE := $(patsubst %_beam,%,$*))
	$(eval MODEL_DIR := $(CURDIR)/models/$(BASE))
	yoyodyne \
		predict \
		--ckpt_path "$(MODEL_DIR)/lightning_logs/version_0/checkpoints/last.ckpt" \
		--data ../../configs/toy_data.yaml \
		--data.model_dir "$(MODEL_DIR)" \
		--data.predict test.tsv \
		--model "../../configs/$(BASE).yaml" \
		--model.beam_width 5 \
		--prediction.path "$@"

clean:
	rm -rf lightning_logs models
