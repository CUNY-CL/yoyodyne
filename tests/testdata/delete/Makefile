SHELL := /bin/bash

all: train.tsv dev.tsv test.tsv

%.tsv:
	paste \
		<(.././generate.py --vocabulary 16 --length 5 --seed 1953 --size 100) \
		/dev/null \
		> $@
