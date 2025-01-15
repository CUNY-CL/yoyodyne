SHELL := /bin/bash

all: train.tsv dev.tsv test.tsv

%.tsv:
	.././generate.py --vocabulary 16 --length 5 --seed 1954 --size 100 > /tmp/identity_$@
	paste /tmp/identity_$@ /tmp/identity_$@ > $@
	rm /tmp/identity_$@
