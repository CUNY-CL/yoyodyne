SHELL := /bin/bash

all: train.tsv dev.tsv test.tsv

%.tsv:
	.././generate.py --vocabulary 16 --length 5 --seed 1957 --size 100 > /tmp/reverse_$@
	paste /tmp/reverse_$@ <(rev /tmp/reverse_$@) > $@
	rm /tmp/reverse_$@
