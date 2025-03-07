SHELL := /bin/bash

all: train.tsv dev.tsv test.tsv

%.tsv:
	.././generate.py --vocabulary 16 --length 5 --seed 1951 --size 100 > /tmp/copy_$@
	paste /tmp/copy_$@ <(paste -d'\0' /tmp/copy_$@ /tmp/copy_$@) > $@
	rm /tmp/copy_$@
