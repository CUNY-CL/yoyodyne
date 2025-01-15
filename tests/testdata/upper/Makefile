SHELL := /bin/bash

all: train.tsv dev.tsv test.tsv

%.tsv:
	.././generate.py --vocabulary 16 --length 5 --seed 1958 --size 100 > /tmp/upper_$@
	paste /tmp/upper_$@ <(tr [:lower:] [:upper:] < /tmp/upper_$@) > $@
	rm /tmp/upper_$@
