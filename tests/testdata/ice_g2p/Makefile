all: train.tsv dev.tsv test.tsv

# Data from the SIGMMORPH 2021 shared task on grapheme-to-phoneme conversion.

train.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/2021-task1/refs/heads/main/data/low/ice_train.tsv > $@ 

dev.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/2021-task1/refs/heads/main/data/low/ice_dev.tsv > $@

test.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/2021-task1/refs/heads/main/data/low/ice_test.tsv > $@
