# Features concatenation

A traditional approach to feature conditioned generation, going back to at least
Kann & Schütze (2016), is to simply concatenate the features to the source
string.

For instance, in a morphological generation task, given the lemma `schließen`
and the [UniMorph](https://unimorph.github.io/) feature bundle
`V;SBJV;PST;2;SG`, one might want the following internal representation:

```python
['s', 'c', 'h', 'l', 'i', 'e', 'ß', 'e', 'n', '[V]', '[SBJV]', '[PST]', '[2]', '[SG]']
```

where `[...]` is used to avoid clashes between source and features symbols.

Yoyodyne itself does not perform this sort of concatenation; rather it uses
model-specific strategies for fusing source and features encodings. Hoewver, if
concatenation is desired, use [`concatenate.py`](concatenate.py) to create a
new TSV file where source and features are concatenated into the source column.

## Usage

For the [SIGMORPHON 2016 shared
task](https://sigmorphon.github.io/sharedtasks/2016/) format:

    source  feat1,feat2,... target

the following would produce a new TSV file with source and features
concatenated:

    paste \
        <(./concatenate.py --features_col 2 --features_sep ',' input.tsv) \
        <(cut -f3 input.tsv) \
        > output.tsv

Alternatively, for the [ConLL-SIGMORPHON 2017 shared
task](https://sigmorphon.github.io/sharedtasks/2017/) format:

    source  target  feat1;feat2;...

the following would have a similar effect:

    paste \
        <(./concatenate.py --features_col 3 input.tsv) \
        <(cut -f2 input.tsv) \
        > output.tsv

Note that with both examples, one should specify `--source_sep ' '` when passing
the resulting data to Yoyodyne.

## References

Kann, K. and Schütze, H. 2016. Single-model encoder-decoder with explicit
morphological representation for reinflection. In *Proceedings of the 54th
Annual Meeting of the Association for Computational Linguistics (Volume 2: Short
Papers)*, pages 555-560.
