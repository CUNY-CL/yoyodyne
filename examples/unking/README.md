# UNKing

A traditional approach to dealing with rare symbols, dating at least back to
Ratnaparkhi (1996), is to replace all symbols below a certain frequency
threshold with a reserved "unknown" symbol, a preprocessing step sometimes
called *UNKing*. One intuition behind this practice is the belief that rare
words in the training set are likely to be distributionally similar to unknown
words in the test data.

Yoyodyne itself does not perform UNKing, but one can use [`unk.py`](unk.py) to
create new TSV files, replacing rare symbols with [the reserved unknown symbol
`<UNK>`](https://github.com/CUNY-CL/yoyodyne/blob/ec39332c838d7d9a51ca5ca9cd1c38c17a13195c/yoyodyne/special.py#L8).
The script also logs the percentage of UNKed symbols in each stream and file.

## Usage

In the following examples, I ignore features; specify a non-zero
`--features_col` argument if they are present.

The following would replace all symbols in the test data that do not occur in
the training or validation data with `<UNK>`:

    ./unk.py \
        --train_input train.tsv --train_output train-unk.tsv \
        --val_input dev.tsv --val_output dev-unk.tsv \
        --test_input test.tsv --test_output test-unk.tsv

Because the frequency threshold defaults to 1, replacements can only occur with
in test with the above example.

    ./unk.py \
        --freq 5 \
        --train_input train.tsv --train_output train-unk.tsv \
        --val_input dev.tsv --val_output dev-unk.tsv \
        --test_input test.tsv --test_output test-unk.tsv

The following would replace all symbols that do not occur at least 5 times in
the training or validation data with `<UNK>`:

The replacements can occur with all three sources in this example.

## References

Ratnaparki, A. 1996. A maximum entropy model for part-of-speech tagging. In
*Conference on Empirical Methods in Natural Language Processing*, pages 133-142.
