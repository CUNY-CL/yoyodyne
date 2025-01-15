all: train.tsv dev.tsv test.tsv

# Data from the CoNLL-SIGMMORPH 2017 shared task on morphological reinflection.

train.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/turkish-train-medium > $@

dev.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/all/task1/turkish-dev > $@

test.tsv:
	curl -s https://raw.githubusercontent.com/sigmorphon/conll2017/refs/heads/master/answers/task1/turkish-uncovered-test > $@
