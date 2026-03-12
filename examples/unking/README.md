# UNKing

A traditional approach to dealing with rare symbols, dating at least back to Ratnaparkhi (1996), is to replace all symbols below a certain frequency threshold with a reserved "unknown" symbol, a preprocessing step sometimes called _UNKing_. The intuition behind UNKing is that rare words in the training set are likely to be distributionally similar to unknown words in the test data.

Yoyodyne itself does not perform UNKing, but one can use [`unk.py`](unk.py) to create new TSV files, replacing rare symbols with [the reserved unknown symbol `<UNK>`](https://github.com/CUNY-CL/yoyodyne/blob/ec39332c838d7d9a51ca5ca9cd1c38c17a13195c/yoyodyne/special.py#L8).

## Usage



The following would replace all symbols that occur only once in the training or validation data with `<UNK>`, 


treating each column separately and assuming separators. It simultaneously replaces any symbol in the test data that does not occur at least once in training or validation with `<UNK>`.

    ./unk.py \
        --train_input train.tsv --train_output train-unk.tsv \
        --val_input val.tsv --val_output val-unk.tsv \
        --test_input test.tsv --test_output test-unk.tsv

To apply UNKing to symbols

It should be straightforward to modify this script to operate on specific columns, rather than all columns.

## References

Ratnaparki, A. 1996. A maximum entropy model for part-of-speech tagging. In _Conference on Empirical Methods in Natural Language Processing_, pages 133-142.
